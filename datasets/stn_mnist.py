import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import elasticdeform

class MNIST_STN_Dataset:
    def __init__(self, root:str, deformation: bool = False):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_set = MyMNIST(root=root, train=True, download=True, transform=transform)
        test_set = MyMNIST(root=root, train=False, download=True, transform=transform)

        if deformation:
            temp_dataset, temp_target = [], []
            for index in range(len(train_set)):
                img, target = train_set.data[index].numpy(), train_set.targets[index]
                img_deformed = elasticdeform.deform_random_grid(img, sigma=25, points=3)
                img_zoomed = elasticdeform.deform_random_grid(img, sigma=25, points=3, zoom=4.0)
                temp_dataset.append(img)
                temp_dataset.append(img_deformed)
                temp_dataset.append(img_zoomed)
                for i in range(3):
                    temp_target.append(target)
            train_set.data = temp_dataset
            train_set.targets = temp_target

            temp_dataset, temp_target = [], []
            for index in range(len(test_set)):
                img, target = test_set.data[index].numpy(), test_set.targets[index]
                img_deformed = elasticdeform.deform_random_grid(img, sigma=25, points=3)
                img_zoomed = elasticdeform.deform_random_grid(img, sigma=25, points=3, zoom=4.0)
                temp_dataset.append(img)
                temp_dataset.append(img_deformed)
                temp_dataset.append(img_zoomed)
                for i in range(3):
                    temp_target.append(target)
            test_set.data = temp_dataset
            test_set.targets = temp_target

        self.train_set = train_set
        self.test_set = test_set


        temp_dataset = []
        for index in range(len(self.train_set)):
            img = self.train_set.data[index]
            img = img = Image.fromarray(np.asarray(img), mode='L')
            img = transform(img)
            temp_dataset.append(img)
        self.train_set.data = temp_dataset

        temp_dataset = []
        for index in range(len(self.test_set)):
            img = self.test_set.data[index]
            img = img = Image.fromarray(np.asarray(img), mode='L')
            img = transform(img)
            temp_dataset.append(img)
        self.test_set.data = temp_dataset

        print(len(self.train_set), len(self.test_set))

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers:int = 0) -> (DataLoader, DataLoader):
        trainloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train)
        testloader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_test)
        return trainloader, testloader


class MyMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # # img = Image.fromarray(img.numpy(), mode='L')
        # img = Image.fromarray(np.asarray(img), mode='L')

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target