import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class MNIST_Dataset:
    def __init__(self, root: str, train_size):
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.train_set = MyMNIST(root=root, train=True, download=True, transform=transform, train_size=train_size)
        self.test_set = MyMNIST(root=root, train=False, download=True, transform=transform, train_size=train_size)

        # print(len(self.train_set), len(self.test_set))

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers:int = 0) -> (DataLoader, DataLoader):
        trainloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        testloader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        return trainloader, testloader

class MyMNIST(MNIST):
    def __init__(self, train_size: int, root: str, train: bool = True, transform = None, download: bool = False, target_transform = None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        if train == True:
            temp_trainset, temp_targets = [], []
            counter = [0 for i in range(10)]
            for index in range(len(self.data)):
                target = self.targets[index]
                if counter[target] < train_size:
                    # img = self.data[index]
                    # img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
                    # self.transform(img)
                    temp_trainset.append(self.data[index])
                    temp_targets.append(target)
                    counter[target] += 1
            self.data = np.asarray(temp_trainset)
            self.targets = np.asarray(temp_targets)
        temp_data, temp_targets = [], []
        for index in range(len(self.data)):
            # img = self.data[index]
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            temp_data.append(img)
            temp_targets.append(target)
        self.data = temp_data
        self.targets = temp_targets


    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # img = Image.fromarray(img.numpy(), mode='L').convert('RGB')

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target