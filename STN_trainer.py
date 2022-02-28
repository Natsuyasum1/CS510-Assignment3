import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import NLLLoss
import numpy as np
import torchvision
import matplotlib.pyplot as plt


class STNTrainer:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    def train(self, train_loader, test_loader, lr, epochs):
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.5e-6)
        criterion = NLLLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_start_time = time.time()

            loss_epoch = 0.0
            n_batches = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.to(torch.float)
                optimizer.zero_grad()

                outputs = criterion(self.model(data), target)
                outputs.backward()
                optimizer.step()

                loss_epoch += outputs.item()
            n_batches += 1
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.6f}\t Test acc: {:6f}'
                .format(epoch+1, epochs, epoch_train_time, loss_epoch/n_batches, self.test(test_loader)))
        train_time = time.time() - start_time
        print('Training time: %.3f' % train_time)
        print('Finished training.')

    def test(self, test_loader):
        self.model.eval()
        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)

                prediction = torch.argmax(outputs, 1)
                correct += (prediction == target).sum().float()
                total += len(target)

            # print('Test Acc: %.6f' % (correct/total).cpu().detach().data.numpy())
        # return correct/total
        return (correct/total).cpu().detach().data.numpy()

    def visualize_stn(self, test_loader):
        def convert_image_np(inp):
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            return inp
        self.model.eval()

        with torch.no_grad():
            data = next(iter(test_loader))[0].to(self.device)
            input_tensor = data.cpu()
            transformed_input_tensor = self.model.stn(data).cpu()

            in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
            out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
        plt.ioff()
        plt.show()
