import extract_face_img as e
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
import cv2 as cv

import numpy as np
import os
import random as r
import os.path
import argparse
from matplotlib import cm

model_path = '/Users/maxshashoua/Documents/Developer/faces/models/2019-09-07'
model_ft = torch.load(model_path)
model_ft.eval()
criterion = nn.CrossEntropyLoss()

crops = e.extractFacesFromScreen('./screenshots/', './screenshots/crops/')
for c in crops:
    d = Image.fromarray(c)
    data_transform = transforms.Compose([
    transforms.Scale(350),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = data_transform(d).float()
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    outputs = model_ft(img)
    _, preds = torch.max(outputs.data, 1)
    print('rating: {}'.format(preds))
    cv.imshow('e',c)
    cv.waitKey(0)
    cv.destroyAllWindows()

