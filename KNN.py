from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
from torch import flatten


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

#trainsize = 3000


# gets two verctores and calculates the euclidean distance between them


def euclidean_distance(v1, v2):
    from math import sqrt
    temp = 0
    for i in range(len(v1)):
        temp += (v1[i]-v2[i])**2
    return sqrt(temp)

# finds the index of the minimum value and returns it


def findmin(A):
    minimum = float('inf')
    for i in range(len(A)):
        if(A[i] < minimum):
            minimum = A[i]
            index = i
    return index


# gets two vectores distances and lables,sorts based on values in distances and permutes the values in lables just like
#permutations in distances
def my_sort(distances, Y_train, k):
    for i in range(k):
        j = findmin(distances[i:])
        distances[i], distances[j+i] = distances[j+i], distances[i]
        Y_train[i], Y_train[j+i] = Y_train[j+i], Y_train[i]

# gets an array and returns the most frequent value in it


def answer(A):
    import collections
    c = collections.Counter(A)
    return c.most_common(1)[0][0]

# calculates knn for a set of test samles and returns a list containing the answer for each test


def knn_predict(test, X_train, Y_train, k):
    answers = []
    for t in test:
        distances = []  # the list of distances between a test and all training data
        y_train = Y_train.copy()
        for sample in X_train:
            distances.append(euclidean_distance(t, sample))
        my_sort(distances, y_train, k)  # sorts the distances ascending
        # finds the most frequent lable between k nearest neighbors and save the answer
        answers.append(answer(y_train[:k]))
    return answers

# a function for calculating number and percentage of errors


def calculate_error(A, B):
    correct = 0
    for i in range(len(A)):
        if(A[i] == B[i]):
            correct += 1
    error = len(A)-correct
    return error, error/len(A)


testsize = 1000

mydataset = CustomImageDataset(
    'annotations_file.csv', 'C:\\Users\\parvi\\Desktop\\Project\\images', data_transform_train)

train_data, test_data = random_split(
    mydataset, [len(mydataset)-testsize, testsize])

print(len(train_data))
print(len(test_data))

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

# print(inputes_train[0])
# print(labels_train.shape)

Y_predict_iris = knn_predict(
    inputes_test[:100], inputes_train, labels_train, 3)
error, error_rate = calculate_error(labels_test[:100], Y_predict_iris)
print('number of errors:', error)
print('error rate :', error_rate)
