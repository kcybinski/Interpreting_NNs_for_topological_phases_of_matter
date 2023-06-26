#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 00:53:04 2022

@author: k4cp3rskiii
"""

import os
import sys
from CNN_testing import model_test
from termcolor import cprint
import pathlib
from datetime import datetime
import humanize
from Models import CNN_Upgrade_ThermEncod
from Disord_realizations_test import test_generalization
from CNN_dev import plot_training_hist
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Begin time
    now_beg = datetime.now()

    # Thermometer encoding levels
    therm_on = True
    levels = 20
    fname = "upgraded_model_x_therm"
    # fname = "upgraded_model_x_therm_checkpoint"
    # fname = "Handpicked/1_upgraded_model_x_therm"

    test_basic_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    test_basic_path_2 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    test_basic_path_3 = pathlib.Path("./Datasets/M=50/W=0.05-500-samples/")
    test_basic_path_4 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    test_disord_path = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
    snum = 20
    total = True

    cprint("[INFO]", "magenta", end=" ")
    cprint("Test dataset path :", "white", end=" ")
    cprint(test_basic_path, "green", end="\n")
    test_dict_basic = model_test(
        plots=False,
        test_dict=True,
        therm=therm_on,
        therm_levels=levels,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name=fname,
        test_path=test_basic_path,
        just_stats=True,
    )
    cprint("[INFO]", "magenta", end=" ")
    cprint("Test dataset path :", "white", end=" ")
    cprint(test_basic_path_2, "green", end="\n")
    test_dict_basic_2 = model_test(
        plots=False,
        test_dict=True,
        therm=therm_on,
        therm_levels=levels,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name=fname,
        test_path=test_basic_path_2,
        just_stats=True,
    )
    cprint("[INFO]", "magenta", end=" ")
    cprint("Test dataset path :", "white", end=" ")
    cprint(test_basic_path_3, "green", end="\n")
    test_dict_basic_3 = model_test(
        plots=False,
        test_dict=True,
        therm=therm_on,
        therm_levels=levels,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name=fname,
        test_path=test_basic_path_3,
        just_stats=True,
    )
    cprint("[INFO]", "magenta", end=" ")
    cprint("Test dataset path :", "white", end=" ")
    cprint(test_basic_path_4, "green", end="\n")
    test_dict_basic_4 = model_test(
        plots=False,
        test_dict=True,
        therm=therm_on,
        therm_levels=levels,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name=fname,
        test_path=test_basic_path_4,
        just_stats=True,
    )

    cprint("[INFO]", "magenta", end=" ")
    cprint("Test dataset path :", "white", end=" ")
    cprint(test_disord_path, "green", end="\n")
    test_dict_disord = model_test(
        plots=True,
        test_dict=True,
        therm=therm_on,
        therm_levels=levels,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name=fname,
        test_path=test_disord_path,
        # just_stats=True,
        sample_num=snum,
    )

    # with open(f"./Models/{fname}_history.pickle", "rb") as f:
    #     H = torch.load(f)
    #     plot_training_hist(H)
    #     plt.show()

    if total:

        cprint("[INFO]", "magenta", end=" ")
        cprint("Generating total stats...", "white", end="\n")

        test_generalization(
            models_tab = [1],
            n_realisations = 5,
            n_slices = 40,
            w_low = 1.0,
            w_high = 4.9,
            v_ticks_num = 11,
            verbose = False,
            save = False,
            show = True,
            model_arch = CNN_Upgrade_ThermEncod,
            model_type = "dumb",
            folder = None,
            name_override=fname,
            therm = therm_on,
            levels = levels,
            )

