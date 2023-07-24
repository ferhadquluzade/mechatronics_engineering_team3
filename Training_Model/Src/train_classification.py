import timm
from timm.utils import accuracy
import random
import argparse
from time import time
import glob
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import DataLoader
import torch.utils.data
import os
from pathlib import Path
from util import misc
from util.misc import NativeScalerWithGradNormCount
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_dict = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 'capsicum': 5, 'carrot': 6, 
                      'cauliflower': 7, 'chilli pepper': 8, 'corn': 9, 'cucumber': 10, 'eggplant': 11, 'garlic': 12, 
                      'ginger': 13, 'grapes': 14, 'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18, 
                      'mango': 19, 'onion': 20, 'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24, 
                      'pineapple': 25, 'pomegranate': 26, 'potato': 27, 'raddish': 28, 'soy beans': 29, 'spinach': 30, 
                      'sweetcorn': 31, 'sweetpotato': 32, 'tomato': 33, 'turnip': 34, 'watermelon': 35}   
class_dict_1 = {'Sugarbeet': 0, 'Charlock': 1, 'Stop': 2, 'Priority': 3}    

@torch.no_grad()
def evaluate(dataloader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_log = misc.MetricLogger(delimiter=" ")
    header = "Test:"
    model = model.to(device)
    #switch to eval mode
    model.eval()

    for batch in metric_log.log_every(dataloader, 10, header=header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking = True)
        target = target.to(device, non_blocking = True)

        output = model(images)
        loss = criterion(output, target)

        output = torch.nn.functional.softmax(output, dim=-1)
        acc1, acc2 = accuracy(output=output, target=target, topk=(1,2))

        batch_size = images.shape[0]
        metric_log.update(loss=loss.item())
        metric_log.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_log.meters['acc2'].update(acc2.item(), n=batch_size)
    metric_log.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_log.acc1, top2=metric_log.acc2, losses = metric_log.loss))

    return {k: meter.global_avg for k, meter in metric_log.meters.items()}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scalar, max_norm=None,
                    log_writer=None, args=None):
    model.train()
    model = model.to(device)
    print_freq = 2
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking = True)
        targets = targets.to(device, non_blocking = True)

        outputs = model(samples)
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(outputs, targets)
        loss /= accum_iter

        loss_scalar(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), 
                    create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_value = loss.item()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("loss is {}, stop training".format(loss_value))
            sys.exit(1)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step/len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('sqnet_train_logs/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('sqnet_train_logs/lr', warmup_lr, epoch_1000x)
            print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss}, Lr: {warmup_lr}")

def build_transform(is_train, args):
    if is_train:
        print("train transform")
        return  torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=0.25),
                torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                torchvision.transforms.ToTensor()
            ]
        )
    else:
        print("eval transform")
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.ToTensor()
            ]
        )

def build_dataset(is_train, args):
    tranform = build_transform(is_train, args=args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test').replace("\\", "/")
    dataset = torchvision.datasets.ImageFolder(path, transform=tranform)
    info = dataset.find_classes(path)
    print(f"find classes from {path} : \t{info[0]}")
    print(f"map classes to index from {path} : \t{info[1]}")
    return dataset

def get_args_parse():
    parser = argparse.ArgumentParser('pre_train', add_help=False)
    parser.add_argument('--mode', default='infer', help='switch train mode and infer mode')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iteration')
    #model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='image h and w')
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--root_path', default='class_dataset')
    parser.add_argument('--output_dir', default='check_points')
    parser.add_argument('--log_dir', default='./classification_logs')
                                    #default='Models/mae_train_model/checkpoint-43.pth
                                    #check_points/checkpoint-19.pth
                                    #check_points/classification_rs18_checkpoint-19.pth
    parser.add_argument('--resume', default='check_points/classification_rs18_checkpoint-19.pth', help='resume from check point')
    parser.add_argument('--start_epoch', default=0, metavar='N', type=int)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_memory', action='store_true', 
                        help='Pin CPU memory in dalaloader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory')
    parser.set_defaults(pin_memory = True) 

    return parser   


def main(args, test_image_path=''):
    global class_dict_1
    mode = args.mode
    print(f'{mode} mode...')

    if (mode == 'train'):
        #construct dataset
        data_train = build_dataset(is_train=True, args=args)
        data_val = build_dataset(is_train=False, args=args)

        sample_train = torch.utils.data.RandomSampler(data_train)

        dataloader_train = DataLoader(dataset=data_train, 
                                      sampler=sample_train, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers, 
                                      pin_memory=args.pin_memory, 
                                      drop_last=False)
        
        dataloader_val = DataLoader(dataset=data_train, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    pin_memory=args.pin_memory, 
                                    drop_last=False)

        #construct model
        #model = timm.create_model(model_name='resnet18', pretrained=True, num_classes=args.num_classes, drop_rate=0.1, drop_path_rate=0.1)
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, args.num_classes, kernel_size=1)
        model.num_classes = args.num_classes

        n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of trainable parameters(M) : {n_parameter / 1.e6}')

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scalar = NativeScalerWithGradNormCount()

        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scalar)

        for epoch in range(args.start_epoch, args.epochs):
            print(f"Epoch: {epoch}")
            print(f"The length of dataloader: {len(dataloader_train)}")

            print("Evaluating...")
            model.eval()
            test_states = evaluate(dataloader=dataloader_val, model=model, device=device)
            print(f"Accuray of network on the {len(data_val)} test images: {test_states['acc1']:.3f}%")

            if log_writer is not None:
                log_writer.add_scalar('sqnet_test_perf/test_acc1', test_states['acc1'], epoch)
                log_writer.add_scalar('sqnet_test_perf/test_acc2', test_states['acc2'], epoch)
                log_writer.add_scalar('sqnet_test_perf/test_loss', test_states['loss'], epoch)
            model.train()

            print("Training...")
            train_one_epoch(model=model, criterion=criterion, data_loader=dataloader_train,
                            optimizer=optimizer, device=device, epoch=epoch+1,
                            loss_scalar=loss_scalar, log_writer=log_writer, args=args)
            
            if args.output_dir:
                print("save check points...")
                misc.save_model(args=args, model=model, model_without_ddp=model, 
                                optimizer=optimizer, loss_scaler=loss_scalar, epoch=epoch)

    else:
        #resnet
        #model = timm.create_model("resnet18", pretrained=True, num_classes=args.num_classes, drop_rate=0.1, drop_path_rate=0.1)

        #squeezenet
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model.classifier[1] = torch.nn.Conv2d(512, args.num_classes, kernel_size=1)
        model.num_classes = args.num_classes

        log_writer = SummaryWriter(log_dir=args.log_dir)

        n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of trainable parameters(M) : {n_parameter / 1.e6}')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scalar = NativeScalerWithGradNormCount()

        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scalar)

        images = glob.glob(os.path.join(args.root_path, 'infer', '*.jpg'))
        infer_num = 0
        for img in images:
            model.eval()
            img = img.replace("\\", "/")
            image = Image.open(img).convert('RGB')
            image = image.resize((args.input_size, args.input_size), Image.ANTIALIAS)
            image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
            ts = time.time()

            with torch.no_grad():
                output = model(image)

            output = torch.nn.functional.softmax(output, dim=-1)
            class_idx = torch.argmax(output, dim=1)[0]
            score = torch.max(output, dim = 1)[0][0]
            te = time.time()
            log_writer.add_scalar('infer/sq_infer_time(ms)', (te-ts)*1000, infer_num)
            infer_num = infer_num + 1

            print(f"image path is : {img}")
            print(f"score is {score.item()}, class id is {class_idx.item()}, class name is {list(class_dict_1.keys())[list(class_dict_1.values()).index(class_idx)]}")


if __name__ == '__main__':
    args = get_args_parse()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #if args.mode == 'train':
    main(args=args)
    # else:
    #     images = glob.glob(os.path.join(args.root_path, 'infer', '*.jpg'))
    #     for img in images:
    #         print('\n')
    #         img = img.replace("\\", "/")
    #         main(args=args, test_image_path=img)
