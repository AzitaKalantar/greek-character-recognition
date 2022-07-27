from torchvision.io import read_image
from torchvision.transforms.functional import get_image_size
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('L')

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            # transforms.ToPILImage(image)
            image = self.transform(image)
        if self.target_transform:
            # transforms.ToPILImage(label)
            label = self.target_transform(label)

        return image, label


def train_model(model, train_data):
    loss_function = nn.CrossEntropyLoss()
    optimizer1 = optim.RMSprop(model.parameters(), lr=0.001)
    for epoch in range(30):  # number of times to loop over the dataset
        current_loss = 0.0
        n_mini_batches = 0
        for i, mini_batch in enumerate(train_data, 0):
            images, labels = mini_batch
            optimizer1.zero_grad()
            outputs = model(images)
            #labels = [torch.stack(label) for label in list(labels)]
            # print(type(outputs))
            loss = loss_function(outputs, labels)
            loss.backward()  # does the backward pass and computes all gradients
            optimizer1.step()  # does one optimisation step
            n_mini_batches += 1
            current_loss += loss.item()  # remember that the loss is a zero-order tensor
        print('Epoch %d loss: %.3f' % (epoch+1, current_loss / n_mini_batches))


def evaluate_model(model, test_data):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#image = read_image("GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B1\saripolos1.jpg")
# print(image.shape)
# print(get_image_size(image))
# print(type(image))


data_transform_train = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])

#trainsize = 3000
testsize = 1000


mydataset = CustomImageDataset(
    'annotations_file.csv', 'C:\\Users\\parvi\\Desktop\\Project\\images\\final_images', data_transform_train)

train_data, test_data = random_split(
    mydataset, [len(mydataset)-testsize, testsize])

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


train_features, train_labels = next(iter(train_dataloader))
"""
img = train_features[0]
label = train_labels[0]
plt.imshow(img.view(32, 32), cmap="gray")
plt.show()
print(f"Label: {label}")
"""


class NN10(nn.Module):

    def __init__(self):
        super(NN10, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 212))  # a linear layer (a matrix, plus biases) with 784 inputs and 10 outputs

    def forward(self, x):  # computes the forward pass ... this one is particularly simple
        x = self.layers(x)
        return x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.layers = nn.Sequential(
            # takes one input channel (greyscale), gives 6 output channes, each from a 3x3 convolutional neuron
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5400, 900),
            nn.ReLU(),
            nn.Linear(900, 212))

    def forward(self, x):  # computes the forward pass ... this one is particularly simple
        x = self.layers(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # takes one input channel (greyscale), gives 6 output channes, each from a 3x3 convolutional neuron
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(6*6*16, 350)
        self.fc2 = nn.Linear(350, 256)
        self.fc3 = nn.Linear(256, 212)

    def forward(self, x):  # computes the forward pass ... this one is particularly simple
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#nn1 = NN1()
model = CNN2()
train_model(model, train_dataloader)
evaluate_model(model, test_dataloader)
evaluate_model(model, train_dataloader)
PATH = './generated/cnn_10.pth'
#torch.save(nn1.state_dict(), PATH)
