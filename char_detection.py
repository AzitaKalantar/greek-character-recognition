# from CNN import load_model
# from extract_char import extract_test_chars_from_bboxs
from matplotlib.pyplot import axis
import torch.nn.functional as F
from PIL import Image, ImageDraw
import pytesseract
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import time

'''
df = df.reset_index()
print(df.head())
draw = ImageDraw.Draw(img)
for index, row in df.iterrows():
    if row["char"] == '~':
        continue
    draw.rectangle([(row["left"], row["top"]), (row[
        "left"]+row["width"], row["top"]+row["height"])])
'''
# model = load_model(path, device)
# print(bboex.head())
# img.save("new.png")
# print(pytesseract.image_to_string(Image.open(
#    "GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B1\saripolos1.jpg"), lang='grc'))


def extract_test_chars_from_bboxs(image, bboxs, char_image_path, classes_train):
    classes_list = []
    char_list = []
    i = 1
    # classes_train = classes_train.squeeze()
    print(classes_train.head())
    # print(classes.head())
    for index, row in bboxs.iterrows():
        im_crop = image.crop(
            (row["left"], row["top"], row["right"], row["bottom"]))

        a = classes_train[classes_train["char"] == row["char"]]
        try:
            cl = int(a["class"])
            ch = str(a["char"])
        except:
            cl = 0
            ch = "a"
            bboxs.drop(axis=0, index=index, inplace=True)
            continue
        classes_list.append(
            cl)
        char_list.append(
            ch
        )
        try:
            im_crop.save(
                char_image_path + "\\{}.jpg".format(i))
            i += 1
        except:
            print("error")
    bboxs["char class"] = classes_list
    bboxs["char shape"] = char_list
    bboxs = bboxs.reset_index()


# bboxs.to_csv("bboxs.csv")
#bboxs = pd.read_csv("bboxs.csv")

#f = bboxs[bboxs["index"] == 241]
# print(int(f["char class"]))

class CustomImageDataset3(Dataset):
    def __init__(self, bboxes, img, transform=None, target_transform=None):
        self.image_ = img
        self.bboxes_ = bboxes
        self.transform = transform
        self.target_transform = target_transform
        print(len(self.bboxes_))

    def __len__(self):
        return len(self.bboxes_)

    def __getitem__(self, idx):
        bbox = self.bboxes_.loc[[idx]]
        #print((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
        char = self.image_.crop((
            int(bbox["left"]), int(bbox["top"]), int(bbox["right"]), int(bbox["bottom"])))
        if self.transform:
            char = self.transform(char)
        return str(bbox["char"]), char


class CustomImageDataset2(Dataset):
    def __init__(self, img_dir, n, transform=None):
        self.img_dir = img_dir
        self.n = n
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx+1)+".jpg")
        image = Image.open(img_path)
        #image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image


class CustomImageDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None, target_transform=None):
        self.img_labels = labels.squeeze()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx+1)+".jpg")
        image = Image.open(img_path)

        label = self.img_labels.iloc[idx]

        # print(label)
        if self.transform:
            # transforms.ToPILImage(image)
            image = self.transform(image)
        if self.target_transform:
            # transforms.ToPILImage(label)
            label = self.target_transform(label)

        return image, label


def predict_and_evaluate(model, model_path, output_class_number, labels, classes_train):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_class_number)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_data = CustomImageDataset(
        labels, char_image_path, data_transform_test)
    test_dataloader = DataLoader(test_data)
    correct = 0
    total = 0
    data_dict = {"predicted": [], "label": []}
    with torch.no_grad():
        i = 0
        for data in test_dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(images.shape)
            # print(images.size())
            images = torch.squeeze(images)
            # print(images.size())
            if predicted != labels:
                i += 1
                t = transforms.ToPILImage()
                image = t(images)
                image.save(
                    "C:\\Users\\parvi\\Desktop\\Project\\wrong\\{}.jpg".format(i))
                # print(data)
                p = classes_train[classes_train["class"] == int(predicted)]
                l = classes_train[classes_train["class"] == int(labels)]
                data_dict["predicted"] = data_dict["predicted"] + \
                    [str(p["char"])]
                data_dict["label"] = data_dict["label"] + [str(l["char"])]
            # print(classes_train.loc[classes_train["class"]
            #                        == int(labels)], int(labels))
            # print(classes_train.loc[classes_train["class"]
            #                        == int(predicted)], int(predicted))
            # print("****************")
            correct += (predicted == labels).sum().item()
    data_fr = pd.DataFrame(data_dict)
    data_fr.to_csv(
        "C:\\Users\\parvi\\Desktop\\Project\\wrong\\wrong.csv", encoding="utf-8")
    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def image_to_bboxs(image_path):
    # Mention the installed location of Tesseract-OCR in your system
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    # Read image from which text needs to be extracted
    img = Image.open(image_path)
    dictionary = pytesseract.image_to_boxes(
        img, output_type=pytesseract.Output.DICT, lang="grc")
    df = pd.DataFrame(dictionary)
    df["top"] = img.size[1]-df["top"]-2
    df["bottom"] = img.size[1]-df["bottom"]+2
    df["right"] = df["right"]+2
    df["left"] = df["left"]-2
    df["width"] = df["right"]-df["left"]
    df["height"] = df["bottom"]-df["top"]
    bboxs = df[["char", "left", "top", "right", "bottom"]]
    bboxs = bboxs[bboxs["char"] != "~"]
    bboxs = bboxs.reset_index()
    return img, bboxs


data_transform_test = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

char_image_path = "C:\\Users\\parvi\\Desktop\\Project\\test_image"


class char_detection_recognition:

    def __init__(self, image_path, Model_classes_Info, model, model_path, output_class_number):
        # Mention the installed location of Tesseract-OCR in your system
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        self.char_image_path = char_image_path
        self.data_transform = data_transform_test
        self.device = device

        self.image_path = image_path
        self.classes_train = pd.read_csv(Model_classes_Info)
        self.model = model
        self.model_path = model_path
        self.output_class_number = output_class_number

        self.img = None
        self.bboxs = None
        self.labels = None

    def image_to_bboxs(self):
        # Read image from which text needs to be extracted
        img = Image.open(self.image_path)
        dictionary = pytesseract.image_to_boxes(
            img, output_type=pytesseract.Output.DICT, lang="grc")
        df = pd.DataFrame(dictionary)
        df["top"] = img.size[1]-df["top"]-2
        df["bottom"] = img.size[1]-df["bottom"]+2
        df["right"] = df["right"]+2
        df["left"] = df["left"]-2
        df["width"] = df["right"]-df["left"]
        df["height"] = df["bottom"]-df["top"]
        bboxs = df[["char", "left", "top", "right", "bottom"]]
        bboxs = bboxs[bboxs["char"] != "~"]
        bboxs = bboxs.reset_index()
        self.img = img
        self.bboxs = bboxs

    def extract_test_chars_from_bboxs(self):
        classes_list = []
        char_list = []
        i = 1
        # classes_train = classes_train.squeeze()
        print(self.classes_train.head())
        # print(classes.head())
        for index, row in self.bboxs.iterrows():
            im_crop = self.img.crop(
                (row["left"], row["top"], row["right"], row["bottom"]))

            a = self.classes_train[self.classes_train["char"] == row["char"]]
            try:
                cl = int(a["class"])
                ch = str(a["char"])
            except:
                cl = 0
                ch = "a"
                self.bboxs.drop(axis=0, index=index, inplace=True)
                continue
            classes_list.append(
                cl)
            char_list.append(
                ch
            )
            try:
                im_crop.save(
                    self.char_image_path + "\\{}.jpg".format(i))
                i += 1
            except:
                print("error")
        self.bboxs["char class"] = classes_list
        self.bboxs["char shape"] = char_list
        self.bboxs = self.bboxs.reset_index()
        self.labels = self.bboxs["char class"]

    def predict_and_evaluate(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_class_number)
        self.model.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        self.model.eval()
        test_data = CustomImageDataset(
            self.labels, char_image_path, data_transform_test)
        test_dataloader = DataLoader(test_data)
        correct = 0
        total = 0
        data_dict = {"predicted": [], "label": []}
        with torch.no_grad():
            i = 0
            for data in test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print(images.shape)
                # print(images.size())
                images = torch.squeeze(images)
                # print(images.size())
                if predicted != labels:
                    i += 1
                    t = transforms.ToPILImage()
                    image = t(images)
                    image.save(
                        "C:\\Users\\parvi\\Desktop\\Project\\wrong\\{}.jpg".format(i))
                    # print(data)
                    p = self.classes_train[self.classes_train["class"] == int(
                        predicted)]
                    l = self.classes_train[self.classes_train["class"] == int(
                        labels)]
                    data_dict["predicted"] = data_dict["predicted"] + \
                        [str(p["char"])]
                    data_dict["label"] = data_dict["label"] + [str(l["char"])]
                # print(classes_train.loc[classes_train["class"]
                #                        == int(labels)], int(labels))
                # print(classes_train.loc[classes_train["class"]
                #                        == int(predicted)], int(predicted))
                # print("****************")
                correct += (predicted == labels).sum().item()
        data_fr = pd.DataFrame(data_dict)
        data_fr.to_csv(
            "C:\\Users\\parvi\\Desktop\\Project\\wrong\\wrong.csv", encoding="utf-8")
        print(
            f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def predic(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_class_number)
        m, _, _ = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(m)
        self.model.eval()
        # test_data = CustomImageDataset2(
        #    char_image_path, len(self.labels), data_transform_test)

        test_data = CustomImageDataset3(
            self.bboxs, self.img, data_transform_test)

        test_dataloader = DataLoader(test_data)
        predictions = []
        tes_label = []
        for data in test_dataloader:
            label, img = data
            # calculate outputs by running images through the network
            output = self.model(img)
            sm = torch.nn.functional.softmax(output.data, 1)
            value, predicted = torch.max(sm, 1)
            #print(value, predicted)
            predicted_char = self.classes_train[self.classes_train["class"] == int(
                predicted)]
            predictions.append(str(predicted_char["char"]))
            tes_label.append(label)
        data_fr = pd.Series(predictions)
        data_fr.to_csv(
            "predictions.csv", encoding="utf-8")
        return data_fr, tes_label


def main():
    image_path = "GRPOLY_Dataset\\GRPOLY-DB-MachinePrinted-B3\\markezinis1.jpg"
    Model_classes_Info = "training_data.csv"
    model = models.resnet18()
    model_path = './generated/resnet_25.pth'
    output_class_number = 212
    c_d_r = char_detection_recognition(
        image_path, Model_classes_Info, model, model_path, output_class_number)
    c_d_r.image_to_bboxs()
    c_d_r.extract_test_chars_from_bboxs()
    since = time.time()
    c_d_r.predic()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    main()
