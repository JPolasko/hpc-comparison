#this file contains modules archutectures and used mapping
import torch
from torch import nn


#ALL  USED COMBINATION HERE
def get_dataset_mapping():
    """
    Returns a list of tuples, each containing the dataset and its associated configuration for model training.
    Each tuple consists of the following elements:

    - dataset_name (str): Name of the dataset (e.g., 'CIFAR10', 'MNIST').
    - model_type (str): Type of the model used (e.g., 'CONV', 'FC', 'RESNET', 'VIT').
    - model_class (class): The class of the model architecture (e.g., `CIFAR_10`, `MNIST_conv`, `CIFAR10_ResNet`).
    - num_classes (int): The number of output classes for the dataset (e.g., 10 for CIFAR10, MNIST).

    Returns:
        list: A list of tuples, each specifying the dataset configuration for the respective model.
    """
    dataset_mapping = (
        #('CIFAR10', 'CONV', CIFAR_10, 10),
        #('MNIST','CONV' , MNIST_conv, 10),
        ('MNIST', 'FC' , MNIST_FC, 10),

    )
    return dataset_mapping


class MNIST_FC(nn.Module):
    """
    """
    def __init__(self, loss):
        super(MNIST_FC, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Linear(784, 500),
                                        nn.Tanh(),
                                        nn.Linear(500, 500),
                                        nn.Tanh(),
                                        nn.Linear(500, 10),
                                        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MNIST_conv(nn.Module):
    def __init__(self, loss):
        super(MNIST_conv, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5),  # 2 or 32 channels
                                        nn.MaxPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 16, kernel_size=5),  # 4 or 64 channels
                                        nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.ReLU(),
                                        nn.Linear(256, 10),  # 64 or 1024 input neurons
                                        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class CIFAR_10(nn.Module):
    """
    """
    def __init__(self, loss):
        super(CIFAR_10, self).__init__()
        self.loss = loss
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.2),

                                        nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.3),

                                        nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.4),

                                        nn.Flatten(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 10)
                                        )

    def forward(self, x):
        x = self.classifier(x)
        return x
