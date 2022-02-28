from sklearn import datasets
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from datasets.mnist import MNIST_Dataset
from datasets.stn_mnist import MNIST_STN_Dataset
from trainer import Trainer
from STN_trainer import STNTrainer
from STN import STN

lr = 0.001
epochs = 100
batch_size = 64
device = torch.device("cuda")


#----------------- ResNet50 setup ------------------
# model = models.resnet50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# num_features =  model.fc.in_features    # number of features of last layer
# model.fc = nn.Linear(num_features, 10)
# model = model.to(device)


# dataset = MNIST_Dataset('./datasets', train_size=90)
# trainloader, testloader = dataset.loaders(batch_size=batch_size)

# trainer = Trainer(model, device)
# trainer.train(train_loader=trainloader, test_loader=testloader, lr=lr, epochs=epochs)


# ----------------- VGG19 setup --------------------
# model = models.vgg19(pretrained=True)
# for param in model.features.parameters():
#     param.requires_grad = False
# num_features = model.classifier[6].in_features
# features = list(model.classifier.children())[:-1]
# features.extend([nn.Linear(num_features, 10)])
# model.classifier = nn.Sequential(*features)

# model = model.to(device)

# dataset = MNIST_Dataset('./datasets', train_size=90)
# trainloader, testloader = dataset.loaders(batch_size=batch_size)

# trainer = Trainer(model, device)
# trainer.train(train_loader=trainloader, test_loader=testloader, lr=lr, epochs=epochs)



# ------------- STN vanilla MNIST setup ---------------
# epochs = 50
# model = STN()
# model = model.to(device)

# dataset = MNIST_STN_Dataset('./datasets', deformation=False)
# trainloader, testloader = dataset.loaders(batch_size=batch_size)

# trainer = STNTrainer(model, device)
# trainer.train(train_loader=trainloader, test_loader=testloader, lr=lr, epochs=epochs)
# trainer.visualize_stn(testloader)


# ------------- STN deformed MNIST setup ---------------
epochs = 50
model = STN()
model = model.to(device)

dataset = MNIST_STN_Dataset('./datasets', deformation=True)
trainloader, testloader = dataset.loaders(batch_size=batch_size)

trainer = STNTrainer(model, device)
trainer.train(train_loader=trainloader, test_loader=testloader, lr=lr, epochs=epochs)
trainer.visualize_stn(testloader)
