from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import datetime

import numpy as np
import os
import random as r
import os.path
import argparse

plt.ion()


imgpath = "/Users/maxshashoua/Documents/Developer/faces/SCUT-FBP5500_v2/Images"
datapath = "/Users/maxshashoua/Documents/Developer/faces/SCUT-FBP5500_v2/train_test_files/split_of_60%training and 40%testing"
    
def makeDatasetList(filename):
    '''
        returns [[filename, rating], [filename, rating], ...]
    '''
    file = open(datapath+'/' + filename, 'r')
    d = {}
    data = []
    for l in file:
        s = l.split()
        d[s[0]] = float(s[1])
    file.close()
    return [[k, d.get(k)] for k in d.keys()]
    # for i in list(d.keys())[:10]:
        # print(i, d.get(i))
        # im = Image.open(imgpath + '/' + i)#.convert('LA')
        # im = transforms.ToTensor()(im).unsqueeze(0)
        # im = torch.autograd.Variable(im)
        # data.append([im, d.get(i)])
    # return data

class RatesDataset(Dataset):
    '''face rating dataset'''
    def __init__(self, dataset_list, img_dir, transform=None):
        """
        args:

        img_dir (string): dir with all imgs
        dataframe (pandas.core.frame.DataFrame): pandas dataframe obtained by read_csv()
        transform (callable, optional): optional transform to be applied on a sample
        """

        self.labels_frame = dataset_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame[idx][0])
        image = Image.open(img_name)
        label = self.labels_frame[idx][1]

        if self.transform:
            image = self.transform(image)

        return [image, label] 



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            since_epoch = time.time()
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for data in dataloaders[phase]:
                #get the inputs
                inputs, labels = data

                inputs = Variable(inputs.type(torch.Tensor))
                labels = Variable(labels.type(torch.LongTensor))
                 # zero the parameter gradients
                optimizer.zero_grad()

                #forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            time_elapsed_epoch = time.time() - since_epoch
            print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, criterion):
    batch_size = dataloaders.get('val').batch_size

    correct = 0
    last_correct = 0
    total = len(dataloaders.get('val')) * batch_size
    tot = 0
    for data in dataloaders.get('val'):
        inputs, labels = data
        inputs = Variable(inputs.type(torch.Tensor))
        labels = Variable(labels.type(torch.LongTensor))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # print(loss.size())
        # print(outputs.size())
        # print(labels.size())
        
        for i in range(len(inputs)):
            # plt.imshow(inputs[i].permute(1, 2, 0))
            # print('pred = {}, actual = {}'.format(preds[i], labels[i]))
            last_correct = correct
            correct += preds[i].item()==labels[i].item()
            # print(correct)
            tot += 1
            if last_correct > correct:
                print('HERE')
            if tot %50 == 0:
                percent = 100*float(correct)/float(tot)
                print('score: {}% , {}/{}, ({} total)'.format(percent, correct, tot, total))
            # input('enter to continue...')
    percent = 100*float(correct)/float(tot)
    print('score: {}% , {}/{}, ({} total)'.format(percent, correct, tot, total))
# model_ft = models.resnet152(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 120)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

tr_data = makeDatasetList('train.txt')#[:30]
ts_data = makeDatasetList('test.txt')#[:10]

train_ds = RatesDataset(tr_data, imgpath, transform=data_transform)
test_ds = RatesDataset(ts_data, imgpath, transform=data_transform)

trainloader = DataLoader(train_ds, batch_size = 4, shuffle=True, num_workers=10)
testloader = DataLoader(test_ds, batch_size=4,
                        shuffle=True, num_workers=10)
datasets = {"train": train_ds, "val": test_ds}
dataloaders = {"train": trainloader, "val": testloader}
parser = argparse.ArgumentParser(description='PyTorch ResNet18 for bumble.')
parser.add_argument("-n", "--new", action="store_true",
                    help="create untrained net")
parser.add_argument("-l", "--load", action="store_true",
                    help="load untrained net from ")
parser.add_argument("-t", "--test", action="store_true",
                    help="test the net")
parser.add_argument("model_path", help="path to model to load")
args = parser.parse_args()
# idx = 29
# print(train_ds[idx][1])
# print("Shape of the image is: ", train_ds[idx][0].shape)
model_ft=None
if args.new:
    model_ft = torchvision.models.resnet18(pretrained=False)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)
    # save model in file
    models_dir = '/Users/maxshashoua/Documents/Developer/faces/models/'
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    path = models_dir + date
    print('Saving model...')
    torch.save(model_ft, path)
    print('Saved model as {}'.format(path))

if args.load:
    model_ft = torch.load(args.model_path)
    model_ft.eval()
if args.test:
    criterion = nn.CrossEntropyLoss()
    test_model(model_ft, dataloaders, criterion)