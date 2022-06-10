import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import cv2
import sys
import torchvision.models as models
import time


class Trainer:
    def __init__(self,images, model, optimizer, batch_size, device):
        self.images_folder = images
        self.classes = {name: i for i, name in enumerate(os.listdir(images))}
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.results = {}
        self._initialize()

    def _initialize(self):
        self.results["Train Loss"] = []
        self.results["Train Accuracy"] = []
        self.results["Validation Loss"] = []
        self.results["Validation Accuracy"] = []

    def train(self, train_loader, valid_loader, epochs):
        for i in range(epochs):
            print(f"EPOCH: {i+1}")
            self.model.train()
            train_loss, train_correct = 0, 0
            for batch_idx, (data, labels) in enumerate(train_loader):
                if self.model == "cuda":
                    data, labels = data.float(), labels.float()
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, labels.long())
                #loss = F.nll_loss(output, labels.long(), reduction='sum')
                predictions = output.max(1, keepdims=True)[1]
                train_correct += predictions.eq(labels.view_as(predictions)).cpu().sum()
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"avg train loss = {train_loss / len(train_loader.dataset)}")
            print(f"train correct = {(train_correct / len(train_loader.dataset)) * 100:.02f}%")
            self.results["Train Accuracy"].append(train_correct.item() / len(train_loader.dataset))
            self.results["Train Loss"].append(train_loss / len(train_loader.dataset))
            valid_loss = 0
            valid_correct = 0

            for data, target in valid_loader:
                if self.device == "cuda":
                    data, target = data.float(), target.float()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                valid_loss += F.cross_entropy(output, target.long()).item()  # calc loss
                prediction = output.max(1, keepdims=True)[1]  # make prediction
                valid_correct += prediction.eq(target.view_as(prediction)).cpu().sum()
            self.results["Validation Accuracy"].append(valid_correct.item() / len(valid_loader.dataset))
            self.results["Validation Loss"].append(valid_loss / len(valid_loader.dataset))  # avg loss for validation this epoch
            print(f"avg valid loss = {valid_loss / len(valid_loader.dataset)}")
            print(f"valid correct = {(valid_correct / len(valid_loader.dataset)) * 100:.02f}%")

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.device == "cuda":
                    data, target = data.float(), target.float()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                prediction = output.max(1, keepdims=True)[1]  # make prediction
                correct += prediction.eq(target.view_as(prediction)).cpu().sum()
        print(f"\n\nTest set:\nAccuracy: {correct}/{len(test_loader.dataset)} {(100. * correct / len(test_loader.dataset)):.0f}%\n")


def initialize(images_folder, classes):
    data = []
    for folder in os.listdir(images_folder):
        for image in os.listdir(f"{images_folder}/{folder}"):
            path = f"{images_folder}/{folder}/{image}"
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            normalized_img = img / 255.0
            normalized_img = np.expand_dims(normalized_img, 0)
            data.append([normalized_img, classes[folder]])

    train_loader, validation_loader, test_loader = partition(data)
    return train_loader, validation_loader, test_loader


def partition(data):
    np.random.shuffle(data)
    data_size = len(data)
    train_test_split = 0.8
    valid_split = 0.1
    valid_size = int(data_size * valid_split)
    train_size = int(train_test_split * data_size)
    test_size = data_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    #print(type(train_dataset), type(test_dataset))

    train_valid_size = train_size - valid_size
    train_data, valid_data = torch.utils.data.random_split(train_dataset, [train_valid_size, valid_size])


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader

BATCH_SIZE = 16


def main():
    images_folder = sys.argv[1]
    classes = {name: i for i, name in enumerate(os.listdir(images_folder))}
    train_loader, validation_loader, test_loader = initialize(images_folder, classes)
    model = models.resnet101(False, (1, 224, 224), num_classes = len(classes))
    #model = VGG(num_classes=len(classes), in_channels=1, architecture='VGG11')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    model.cuda()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(images_folder, model, optimizer, BATCH_SIZE, device)
    epochs = 15
    trainer.train(train_loader, validation_loader, epochs)
    trainer.test(test_loader)


    should_save = True
    if should_save:
        torch.save(model.state_dict(), "ML/model.pt")



def main2():

    model_path = "ML/model.pt"
    images_folder = sys.argv[1]
    classes = {name: i for i, name in enumerate(os.listdir(images_folder))}
    print(classes)
    train_loader, validation_loader, test_loader = initialize(images_folder, classes)

    model = models.resnet101(False, (1, 224, 224), num_classes=len(classes))
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr)
    device = torch.device("cuda")
    model.to(device)

    trainer = Trainer(images_folder, model, optimizer, BATCH_SIZE, device)
    t = time.time()
    trainer.test(test_loader)
    t2 = time.time()
    num_samples = len(test_loader.dataset)
    print(f"time passed: {t2 - t} for {num_samples} examples\n-> time per example: {(t2-t) / num_samples}")

if __name__ == "__main__":
    main()