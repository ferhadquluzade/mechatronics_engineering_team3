import torchvision.transforms as transforms
from xy_dataset import XYDataset, XYDataset_static
from torchvision.models.resnet import ResNet18_Weights
import torch
import torchvision
import time
import torch.nn.functional as F
import PIL
from IPython.display import display

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.utils.data
import glob
import os

TASK = 'road_following'

CATEGORIES = ['apex']

DATASETS = ['A', 'B']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(distortion_scale=0.6, p=0),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dir_path = "D:/DLFile/jetracer_notebook/dataset/train/apex" 
test_dir_path = "D:/DLFile/jetracer_notebook/dataset/test/apex" 

train_dirs = glob.glob(os.path.join(train_dir_path, '*.jpg'))
test_dirs = glob.glob(os.path.join(test_dir_path, '*.jpg'))

train_dataset = XYDataset_static(train_dirs, CATEGORIES, transform=TRANSFORMS, random_hflip=True)
test_dataset = XYDataset_static(test_dirs, CATEGORIES, transform=TRANSFORMS, random_hflip=True)

device = torch.device('cuda')
output_dim = 2  # x, y coordinate for each category

# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, output_dim)

# SQUEEZENET 
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, output_dim, kernel_size=1)
# model.num_classes = len(dataset.categories)

# RESNET 18
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(512, output_dim)

# RESNET 34
#model = torchvision.models.resnet34(pretrained=True)
#model.fc = torch.nn.Linear(512, output_dim)

model = model.to(device)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

BATCH_SIZE = 8

optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
log_writer = SummaryWriter("training_logs")

epochs = 30
global_step = 1
print_freq = 20

def train_eval(is_training):
    global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, train_dataset, optimizer, global_step
    
    sample_train = torch.utils.data.RandomSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        sampler=sample_train,
        batch_size=BATCH_SIZE,
        #shuffle = True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    time.sleep(0.5)

    if is_training:
        print('training...')
        model = model.train()
    else:
        print('eval...')
        model = model.eval()

    for epoch in range(epochs):
        
        i = 0
        print("----------{} th epoch training start----------".format(epoch+1))
        sum_loss = 0.0
        error_count = 0.0
        for images, category_idx, xy in iter(train_loader):
            # send data to device
            images = images.to(device)
            xy = xy.to(device)

            if is_training:
                # zero gradients of parameters
                optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)
            # compute MSE loss over x, y coordinates for associated categories
            loss = 0.0
            for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                output = outputs[batch_idx][2 * cat_idx:2 * cat_idx+2]
                target = xy[batch_idx]
                loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx+2] - xy[batch_idx])**2)
            loss /= len(category_idx)

            if is_training:
                # run backpropogation to accumulate gradients
                loss.backward()

                # step optimizer to adjust parameters
                optimizer.step()

            # increment progress
            count = len(category_idx.flatten())
            i += count
            sum_loss += float(loss)
            if global_step % print_freq == 0:
                print("training steps: {}, Loss is: {}".format(global_step, sum_loss / i))
            #log_writer.add_scalar("train_loss/resnet_34", sum_loss / i, global_step)
            global_step += 1

        print("Evaluating...")
        sum_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, category_idx, xy in iter(test_loader):
                images = images.to(device)
                xy = xy.to(device)

                outputs = model(images)

                loss = 0.0
                for batch_idx, cat_idx in enumerate(list(category_idx.flatten())):
                    loss += torch.mean((outputs[batch_idx][2 * cat_idx:2 * cat_idx+2] - xy[batch_idx])**2)
                loss /= len(category_idx)
                sum_loss += float(loss)
        print(f"The average test loss of current epoch is :{sum_loss / len(test_dataset)}")
        #log_writer.add_scalar("test_loss/resnet_34", sum_loss / len(test_dataset), epoch + 1)

        model.train()


        if epoch >= 9:
            torch.save(model.state_dict(), "D:/DLFile/jetracer_notebook/checkpoints/checkpoints_rs18_{}_epoch.pth".format(epoch+1))
            print("model saved!")

    model = model.eval()

    #state_widget.value = 'live'

    

train_eval(is_training=True)