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
import time


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

testsize = 500

train_data = CustomImageDataset(
    'annotations_file.csv', 'C:\\Users\\parvi\\Desktop\\Project\\images\\final_images', data_transform_train)


test_data = CustomImageDataset(
    'annotations_file_test.csv', 'C:\\Users\\parvi\\Desktop\\Project\\images\\final_test', data_transform_train)


train_dataloader = DataLoader(
    train_data, shuffle=True, batch_size=len(train_data))

test_dataloader = DataLoader(
    test_data, shuffle=True, batch_size=len(test_data))

inputes_train, labels_train = next(iter(train_dataloader))

inputes_test, labels_test = next(iter(test_dataloader))


inputes_train = flatten(inputes_train, start_dim=1)
inputes_test = flatten(inputes_test, start_dim=1)


inputes_train = inputes_train.numpy()
labels_train = labels_train.numpy()
inputes_test = inputes_test.numpy()
labels_test = labels_test.numpy()

"""
svm_rbf_05 = svm.SVC(kernel='rbf', C=0.5, decision_function_shape="ovr")
m = svm_rbf_05.fit(inputes_train, labels_train)
predictions = svm_rbf_05.predict(inputes_test)
accuracy_5_test = metrics.accuracy_score(
    y_true=labels_test, y_pred=predictions)
predictions = svm_rbf_05.predict(inputes_train)
accuracy_5_train = metrics.accuracy_score(
    y_true=labels_train, y_pred=predictions)
print(accuracy_5_test)
print(accuracy_5_train)

print("--------")

svm_rbf_10 = svm.SVC(kernel='rbf', C=1, decision_function_shape="ovr")
m = svm_rbf_10.fit(inputes_train, labels_train)
predictions_test = svm_rbf_10.predict(inputes_test)
accuracy_10_test = metrics.accuracy_score(
    y_true=labels_test, y_pred=predictions_test)
predictions_train = svm_rbf_10.predict(inputes_train)
accuracy_10_train = metrics.accuracy_score(
    y_true=labels_train, y_pred=predictions_train)
print(accuracy_10_test)
print(accuracy_10_train)

print("--------")

svm_rbf_50 = svm.SVC(kernel='rbf', C=5, decision_function_shape="ovr")
m = svm_rbf_50.fit(inputes_train, labels_train)
predictions = svm_rbf_50.predict(inputes_test)
accuracy_50_test = metrics.accuracy_score(
    y_true=labels_test, y_pred=predictions)
predictions = svm_rbf_50.predict(inputes_train)
accuracy_50_train = metrics.accuracy_score(
    y_true=labels_train, y_pred=predictions)
print(accuracy_50_test)
print(accuracy_50_train)

print("--------")

"""

svm_rbf_100 = svm.SVC(kernel='rbf', C=10,
                      decision_function_shape="ovr", probability=True)
m = svm_rbf_100.fit(inputes_train, labels_train)
since = time.time()
predictions = svm_rbf_100.predict(inputes_test)
prediction_prob = svm_rbf_100.predict_proba(inputes_test)

for i in range(5):
    print(predictions[i])
    print(prediction_prob[i][predictions[i]])
    print("----")
accuracy_100_test = metrics.accuracy_score(
    y_true=labels_test, y_pred=predictions)
predictions = svm_rbf_100.predict(inputes_train)
accuracy_100_train = metrics.accuracy_score(
    y_true=labels_train, y_pred=predictions)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print(accuracy_100_test)
print(accuracy_100_train)

# Training complete in 0m 48s
# 0.9448818897637795
# 0.9996522948539638
# 22s
"""

plt.plot([0.5, 1, 5, 10], [accuracy_5_train, accuracy_10_train,
                           accuracy_50_train, accuracy_100_train], '-o')

plt.plot([0.5, 1, 5, 10], [accuracy_5_test, accuracy_10_test,
                           accuracy_50_test, accuracy_100_test], '-o')

plt.legend(["Train", "Test"])
plt.xlabel("C")
plt.ylabel("Test Accuracy")
plt.show()


0.8926270579813886
0.8473574408901252
--------
0.9284180386542591
0.9391515994436718
--------
0.9463135289906943
0.9965229485396384
--------
0.9448818897637795
0.9996522948539638
"""
