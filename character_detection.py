import os
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from math import ceil, floor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from bs4 import BeautifulSoup


class MyDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, "Xmls"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        xml_path = os.path.join(self.root, "Xmls", self.xmls[idx])
        img = Image.open(img_path).convert("RGB")
        xml = open(xml_path, "r", encoding="utf8")
        contents = xml.read()
        soup = BeautifulSoup(contents, 'xml')
        glyphs = soup.find_all('Glyph')
        boxes = []
        labels = []
        lables_set = []
        for glyph in glyphs:
            points = []
            points_str = str(glyph.Coords['points']).split(" ")
            for point in points_str:
                points.append(tuple(map(int, point.split(','))))

            if len(points) != 4 or len(glyph.get_text()) < 4 or points[0][0] > points[2][0] or points[0][1] > points[2][1]:
                continue

            xmin = floor(points[0][0])
            xmax = ceil(points[2][0])
            ymin = points[0][1]
            ymax = points[2][1]

           # boxes.append([int((xmin/2500)*500), int((ymin/3500)*700),
            #             int((xmax/2500)*500), int((ymax/3500)*700)])
            boxes.append([xmin, ymin,
                         xmax, ymax])
            # labels.append(1)
            if not glyph.get_text()[3] in lables_set:
                lables_set.append(glyph.get_text()[3])
            labels.append(lables_set.index(glyph.get_text()[3])+1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # img = img.resize((500, 700))
        # print(img.size)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                  hidden_layer,
    #                                                 num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 126
    # use our dataset and defined transformations
    dataset = MyDataset(
        'GRPOLY_Dataset\\GRPOLY-DB-MachinePrinted-B1\\', get_transform(train=True))
    dataset_test = MyDataset(
        'GRPOLY_Dataset\\GRPOLY-DB-MachinePrinted-B1\\', get_transform(train=False))

    print(len(dataset))
    print(len(dataset_test))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    print(indices)
    dataset = torch.utils.data.Subset(dataset, indices[:-1])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1:])
    print(len(dataset))
    print(len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    # pick one image from the test set
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    print(prediction[:10])


if __name__ == "__main__":
    main()
