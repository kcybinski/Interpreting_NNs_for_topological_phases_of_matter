#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:15:05 2022

@author: k4cp3rskiii
"""


import torch.nn as nn
from torch import flatten


#%%
class CNN_SSH_Ania(nn.Module):
    def __init__(self):
        super(CNN_SSH_Ania, self).__init__()

        # W_out = (W_input - filter_size + 2*padding) / stride + 1

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),  # , stride=2, padding=2, bias=False),
            # W_out = 46
            nn.MaxPool2d((2, 2)),
            # W_out = 23
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Conv2d(5, 8, kernel_size=4),  # stride=2, padding=2, bias=False),
            # W_out = 20
            nn.MaxPool2d((2, 2)),
            # W_out = 10
            # nn.BatchNorm2d(3),
            nn.ReLU()
            # nn.Dropout(p=0.5),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(8, 15, kernel_size=3),
            # W_out = 8
            # nn.BatchNorm2d(7),
            nn.ReLU(),
        )
        self.avg_pool = nn.Sequential(nn.AvgPool2d(8))

        self.linear_layers = nn.Sequential(
            nn.Linear(15, 2)
            # nn.Dropout(p=0.3),
            # nn.Linear(15, 1),
        )

    def forward(self, x):
        # print('init',x.shape)
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        # print('before pooling',x.shape)
        x = self.avg_pool(x)
        # print('after pooling',x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        # self.float()
        # print('after linear',x.shape)
        return x


class CNN_SSH_Basic(nn.Module):
    def __init__(self):
        super(CNN_SSH_Basic, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(2, 3, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(3, 4, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(4, 5, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.MaxPool2d((2,2)),
            nn.Conv2d(5, 6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(inplace=True)
            # nn.MaxPool2d((2,2)),
        )

        self.cnn_layers_2 = nn.Sequential(  # unikać replicate
            nn.Conv2d(6, 7, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(inplace=True),
        )
        # nn.MaxPool2d((2,2))
        # po tym powinno działać  - hak zaczepia się na macierzy, a nie pojedynczej wartości
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            )  # potrzebny do cama pool - nie trzeba za dużo definiować
            # ale z nim coś mocno nie działa
        )

        self.linear_layers = nn.Sequential(  # 80x80 do 12x12
            # printować problematyczne obrazki
            nn.Linear(7, 2),  # TU MOŻNABY 2 KLASY OUT ZAMIAST 1?
            # nn.Dropout(p=0.3),
            # nn.Linear(15, 1),
            # nn.Sigmoid()#dim=1) #czy jest potrzebbny - czy bceloss ma go wbudowanego - cross entropy ma już wbudowane
        )

    def forward(self, x):
        # print('init',x.shape)
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        # print('after cnn',x.shape)
        x = self.avg_pool(x)
        # print('after GAP',x.shape)
        x = x.view(x.size(0), -1)  # - niepotrzebne przy global avg
        # print(x.shape)
        x = self.linear_layers(x)
        # self.float()
        # print('after linear',x.shape)
        return x


# =============================================================================
#
# =============================================================================


class CNN_Upgrade(nn.Module):
    def __init__(self):
        super(CNN_Upgrade, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4, stride=2, padding=2, bias=False),
            # W_out = 26
            nn.BatchNorm2d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(5, 7, kernel_size=3, stride=1, padding=1, bias=False),
            # W_out = 26
            nn.BatchNorm2d(7),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.MaxPool2d((2,2)),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(7, 9, kernel_size=3, stride=1, padding=1, bias=False),
            # W_out = 26
            nn.BatchNorm2d(9),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9, 2), nn.Softmax(dim=1))

    def forward(self, x):
        # print('init',x.shape)
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        # print('after cnn',x.shape)
        x = x.view(x.size(0), -1)  # - niepotrzebne przy global avg
        # print(x.shape)
        x = self.linear_layers(x)
        # self.float()
        # print('after linear',x.shape)
        return x


# =============================================================================
#
# =============================================================================


class CNN_Upgrade_ThermEncod(nn.Module):
    def __init__(self, levels, in_layer_1 = 1, out_layer_1=7, out_layer_2 = 9):
        super(CNN_Upgrade_ThermEncod, self).__init__()
        self.in_layer_1 = in_layer_1
        self.out_layer_1 = out_layer_1
        self.out_layer_2 = out_layer_2
        

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_layer_1*levels, 
                      out_channels=5*levels, 
                      kernel_size=4, 
                      stride=2, 
                      padding=2, 
                      bias=False),
            # W_out = ___ * levels
            nn.BatchNorm2d(5*levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(5*levels, 7*levels, kernel_size=3, stride=1, padding=1, bias=False),
            # W_out = ___ * levels
            nn.BatchNorm2d(7*levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.MaxPool2d((2,2)),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(7*levels, 9*levels, kernel_size=3, stride=1, padding=1, bias=False),
            # W_out = ___
            nn.BatchNorm2d(9*levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9*levels, 2), nn.Softmax(dim=1))

    def forward(self, x):
        # print('init',x.shape)
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        # print('after cnn',x.shape)
        x = x.view(x.size(0), -1)  # - niepotrzebne przy global avg
        # print(x.shape)
        x = self.linear_layers(x)
        # self.float()
        # print('after linear',x.shape)
        return x


# =============================================================================
#
# =============================================================================


class CNN_SSH_Kacper(nn.Module):
    def __init__(self):
        super(CNN_SSH_Kacper, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=6, stride=2, padding=2, bias=False),
            # W_out = 25
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(2, 3, kernel_size=5, stride=2, padding=2, bias=False),
            # W_out = 13
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=1, bias=False),
            # W_out = 11
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(4, 5, kernel_size=5, stride=1, padding=1, bias=False),
            # W_out = 9
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.MaxPool2d((2,2)),
            nn.Conv2d(5, 6, kernel_size=3, stride=1, padding=1, bias=False),
            # W_out = 9
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d((2,2)),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(6, 7, kernel_size=4, stride=1, padding=1, bias=False),
            # W_out = 8
            nn.BatchNorm2d(7),
            nn.ReLU(inplace=True),
        )
        # nn.MaxPool2d((2,2))

        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(2, padding=0, stride=1)
            # W_out = 7
        )

        self.linear_layers = nn.Sequential(  # 80x80 do 12x12
            # printować problematyczne obrazki
            nn.Linear(343, 2),  # TU MOŻNABY 2 KLASY OUT ZAMIAST 1?
            # nn.Dropout(p=0.3),
            # nn.Linear(15, 1),
            nn.Sigmoid(),  # dim=1) #czy jest potrzebbny - czy bceloss ma go wbudowanego - cross entropy ma już wbudowane
        )

    def forward(self, x):
        # print('init',x.shape)
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        # print('after cnn',x.shape)
        x = x.view(x.size(0), -1)  # - niepotrzebne przy global avg
        # print(x.shape)
        x = self.linear_layers(x)
        # self.float()
        # print('after linear',x.shape)
        return x


# =============================================================================
#
# =============================================================================


class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(
            in_channels=numChannels, out_channels=20, kernel_size=(5, 5)
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
