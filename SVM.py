import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from torch import flatten
from sklearn import metrics


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


#image = read_image("GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B1\saripolos1.jpg")
# print(image.shape)
# print(get_image_size(image))
# print(type(image))
data_transform_train = transforms.Compose([
    transforms.Resize([20, 20]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])

testsize = 1000

mydataset = CustomImageDataset(
    'annotations_file.csv', 'C:\\Users\\parvi\\Desktop\\Project\\images', data_transform_train)

train_data, test_data = random_split(
    mydataset, [len(mydataset)-testsize, testsize])


train_dataloader = DataLoader(
    train_data, shuffle=True, batch_size=len(train_data))

test_dataloader = DataLoader(
    test_data, shuffle=True, batch_size=len(test_data))

inputes_train, labels_train = next(iter(train_dataloader))

inputes_test, labels_test = next(iter(train_dataloader))

inputes_train = flatten(inputes_train, start_dim=1)
inputes_test = flatten(inputes_test, start_dim=1)


inputes_train = inputes_train.numpy()
labels_train = labels_train.numpy()
inputes_test = inputes_test.numpy()
labels_test = labels_test.numpy()


svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(inputes_train, labels_train)
predictions = svm_rbf.predict(inputes_test)
print(metrics.accuracy_score(y_true=labels_test, y_pred=predictions))
