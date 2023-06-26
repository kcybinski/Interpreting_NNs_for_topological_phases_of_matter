#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:04:19 2023

@author: k4cp3rskiii
"""

import warnings

warnings.filterwarnings("ignore")

# Pytorch imports
import torch
import sys

import torch.optim as optim
from termcolor import cprint
import matplotlib.pyplot as plt

import pathlib
from datetime import datetime
import humanize

from Loaders import Importer
from Models import CNN_Upgrade_ThermEncod
from CNN_dev import train_model, test_model, plot_training_hist
from CAM_Sal_viz_utils import models_viz_datagen, CAMs_Sal_Viz
from CNN_testing import model_test

# %%

if __name__ == "__main__":
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ds_clean = pathlib.Path("./Datasets/M=50/1000-200-100/")
    # ds_clean = pathlib.Path("./Datasets/M=50/2000-400-200/")
    ds_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    ds_w_001 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    ds_w_005 = pathlib.Path("./Datasets/M=50/W=0.05-500-samples/")
    ds_w_015 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    ds_w_050 = pathlib.Path("./Datasets/M=50/W=0.50-500-samples/")
    ds_w_100 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
    ds_w_300 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")

    # Model training
    if sys.platform.startswith("darwin"):
        def_batch_size = 500
        epochs = 500
    else:
        def_batch_size = 100
        epochs = 150
    # Thermometer encoding levels
    therm_on = True
    levels = 10
    fname = "upgraded_model_x_therm"
    lr = 0.001
    # lr = 0.1
    # moment = 0.99
    moment = 0.95
    weight_decay = 0.1
    # weight_decay = 0
    gamma = 0.1
    noise_pow = 0
    es_dict = {"patience": 5, "tolerance": 0.000001, "warmup": 50}
    # es_dict = None
    ds_train = ds_clean
    ds_viz = ds_clean
    perturb_list = [ds_w_001, ds_w_005, ds_w_050]
    # perturb_list = None
    # ==============================================================================
    # Here we only have two classes, defined by Winding number:
    # Winding number = 0
    # Winding number = 1
    # ==============================================================================
    num_classes = 2
    class_names = ["Trivial", "Topological"]

    # %%

    ds = Importer(
        ds_train,
        def_batch_size,
        device,
        therm_levels=levels,
        perturbative_ds=perturb_list,
    )
    train_loader = ds.get_train_loader(seed=2137)
    val_loader = ds.get_val_loader()
    test_loader = ds.get_test_loader()

    model = CNN_Upgrade_ThermEncod(levels)

    model = model.float()
    model.to(device)
    # optimizer = optim.SGD(
    #     model.parameters(), lr=lr, momentum=moment, weight_decay=weight_decay
    # )
    # optimizer = optim.SGD(
    #     model.parameters(), lr=lr, momentum=moment, weight_decay=weight_decay
    # )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=gamma, patience=5)
    # scheduler=None

    now_beg = datetime.now()

    train_history = train_model(
        model,
        epochs,
        train_loader,
        val_loader,
        optimizer,
        def_batch_size,
        num_classes,
        device,
        print_all=True,
        therm=therm_on,
        therm_levels=levels,
        checkpoint_best=fname,
        early_stopping=es_dict,
        scheduler=scheduler,
    )
    # test_dict = test_model(model, test_loader, class_names, device, therm=therm_on, therm_levels=levels)
    # plot_training_hist(train_history)

    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint(" Total elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")

    # serialize the model to disk
    torch.save(model.state_dict(), f"Models/{fname}.dict")
    torch.save(train_history, f"Models/{fname}_history.pickle")

# %%
"""
    now_beg = datetime.now()
    model_test(plots=False, sample_num=0, cams_data_out=False,
               therm=therm_on, therm_levels=levels,
               model_arch=CNN_Upgrade_ThermEncod,
               model_name=fname,
               test_path=ds_train)
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint(" Total elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")

#%%

    # If 'None', then all dataset is processed
    samples=10

    now_beg = datetime.now()

    data_viz = models_viz_datagen(samples_num=samples,
                                  model_no=1,
                                  autogenerate=False,
                                  model_name_override=fname,
                                  model_arch=CNN_Upgrade_ThermEncod,
                                  ds_path=ds_viz,
                                  therm=therm_on,
                                  therm_levels=levels,
                                  verbose=True)

    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint(" Total elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")

    class_names = ["Trivial", "Topological"]
    # ind = 5

    sal_tab, CAMs_tab, pred_tab = data_viz

#%%
    for ind in range(10):
        fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))
        CAMs_Sal_Viz(fig_cam, ax_cam, CAMs_tab[ind], sal_tab[ind], ds_viz,
                     pred_tab[ind], class_names, cam_img_num=ind,
                     model_name=f"{fname}", M=50)

"""
