#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 07:08:23 2023

@author: k4cp3rskiii
"""

import torch
import pathlib
# from Loaders import Importer
import matplotlib.pyplot as plt
import seaborn as sns

def one_hot(x, levels):
    """
    Output
    ------
    One hot Encoding of the input.
    """

    batch_size, channel, H, W = x.size()
    x = x.unsqueeze_(4)
    x = torch.ceil(x * (levels-1)).long()
    onehot = torch.zeros(batch_size, channel, H, W, levels).float().scatter_(4, x, 1)

    return onehot

def one_hot_to_thermometer(x, levels):
    """
    Convert One hot Encoding to Thermometer Encoding.
    """

    thermometer = torch.ones(size=x.shape) - torch.cumsum(x , dim = 4)

    return thermometer

def Thermometer(x, levels):
    """
    Output
    ------
    Thermometer Encoding of the input.
    """

    onehot = one_hot(x, levels)

    thermometer = one_hot_to_thermometer(onehot, levels)

    return thermometer

def data_to_therm(x, levels):
    # Application of thermometer encoding
    encoding = Thermometer(x, levels)
    # These 3 lines below recode the 'levels' dimension into tensor channels
    encoding = encoding.permute(0, 2, 3, 1, 4)
    encoding = torch.flatten(encoding, start_dim = 3)
    encoding = encoding.permute(0, 3, 1, 2)
    return encoding

def data_to_therm_numpy(x, levels):
    x = torch.tensor(x)
    # Application of thermometer encoding
    encoding = Thermometer(x, levels)
    # These 3 lines below recode the 'levels' dimension into tensor channels
    encoding = encoding.permute(0, 2, 3, 1, 4)
    encoding = torch.flatten(encoding, start_dim = 3)
    encoding = encoding.permute(0, 3, 1, 2)
    return encoding.numpy()



if __name__ == "__main__":
    """
    levels = 100
    ds_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    ds_w_001 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    ds_w_015 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    ds_w_100 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
    ds_w_300 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")

    ds_choice = ds_clean
    def_batch_size = 250
    ind = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          "cpu")
    ds = Importer(ds_choice, def_batch_size, device)
    test_loader = ds.get_test_loader()
    x = test_loader.dataset.X
    y = test_loader.dataset.Y
    class_names = ["Trivial", "Topological"]
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.reshape([-1, 1, 50, 50])
        if batch_idx == 0:
            th = Thermometer(x, levels)

    th_np = th.numpy()
    for lr_num, layer in enumerate(range(levels)):
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(th_np[ind, 0, :, :, layer], ax=ax, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{str(ds_choice)}\n ind={ind} | {class_names[y[ind]]}\nChannel {lr_num}")
        p = pathlib.Path("./ThermEncod_demo/")
        plt.savefig(p.joinpath(f"channel_{lr_num}.pdf"))
        plt.show()

    view_th = th[1, 0, :, :, :].numpy()
    """
