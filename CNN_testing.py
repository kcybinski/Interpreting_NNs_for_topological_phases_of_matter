#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 02:16:42 2022

@author: k4cp3rskiii
"""

import warnings

import humanize
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint

warnings.filterwarnings("ignore")

# Pytorch imports
import torch

import pathlib
from datetime import datetime

from Loaders import Importer
from Models import CNN_Upgrade, CNN_Upgrade_ThermEncod
from CNN_dev import (
    get_preds_from_model,
    test_model,
    get_CAM,
    CAMs_viz,
)


def model_test(
    plots=True,
    test_dict=False,
    test_path=None,
    cams_data_out=False,
    sample_num=None,
    model_name=None,
    therm=False,
    therm_levels=100,
    verbose=True,
    model_arch=None,
    just_stats=False,
    just_stats_and_preds=False,
    realization=None,
):
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if test_path is None:
        # ds_path = pathlib.Path("./Datasets/M=50/1000-200-100/")
        # ds_path = pathlib.Path("./Datasets/M=50/2000-400-200/")
        # ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
        ds_path = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
    else:
        ds_path = test_path
    def_batch_size = 100

    # ==============================================================================
    # Here we only have two classes, defined by Winding number:
    # Winding number = 0
    # Winding number = 1
    # ==============================================================================
    num_classes = 2
    class_names = ["Trivial", "Topological"]

    if model_name is None:
        # model_name = 'working_model_8_best_so_far'
        # model_name = 'working_model_9'

        # model_name = "upgraded_model_upper_edge_state"
        # model_name = 'upgraded_model_good_not_edge'
        model_name = "upgraded_model_x"
        # model_name = 'dumb_model_autogenerate_1'
        # model_name = 'upgraded_model_autogenerate_2'

        # Model with architecture CNN_SSH_Kacper
        # model_name = 'working_model_11'

    if therm:
        ds = Importer(ds_path, def_batch_size, therm_levels=therm_levels)
    else:
        ds = Importer(ds_path, def_batch_size)

    test_loader = ds.get_test_loader(realization=realization)
    # model = torch.load(f'Models/{model_name}.pt').float().to(device)

    CAMs_tab = []

    if sample_num is None:
        sample_num = test_loader.dataset.Y.shape[0]

    for img_idx in range(max(sample_num, 1)):

        # Idk why, but moving these lines out of the loop resulted in the program clogging up gigabytes of RAM and
        # crashing computer

        if model_arch is None:
            model = CNN_Upgrade()
        else:
            model = model_arch(therm_levels)
        model.load_state_dict(
            torch.load(f"Models/{model_name}.dict", map_location=torch.device("cpu"))
        )
        model = model.float().to(device)

        preds = get_preds_from_model(
            model,
            test_loader,
            class_names,
            device,
            therm=therm,
            therm_levels=therm_levels,
        )

        if just_stats:
            test_out = test_model(
                model, test_loader, class_names, device, verbose=verbose
            )
            return test_out

        if just_stats_and_preds:
            test_out = test_model(
                model, test_loader, class_names, device, verbose=verbose
            )
            return test_out, preds, test_loader.dataset.Y

        if sample_num == 0:
            test_out = test_model(
                model, test_loader, class_names, device, verbose=verbose
            )
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            fig.suptitle(
                f"Model : {model_name} | W = {np.unique(ds.W_tab)[0]}", fontsize=20
            )
            plt.tight_layout()
            ax.plot(ds.v_tab, ds.test_labels, lw=0, marker=6, label="True label")
            ax.plot(ds.v_tab, preds, lw=0, marker=7, label="Predicted label")
            ax.set_ylabel("Winding Number $\mathcal{W}$", fontsize=18)
            ax.set_xlabel("$\mathcal{v}$", fontsize=18)
            ax.set_yticks([0, 1])
            ax.axvline(x=1.0, ls="dotted")
            plt.legend(loc="lower left", fontsize=25)
            break

        cam_img_num = img_idx

        CAMs_topo, CAMs_trivial = get_CAM(
            model,
            ds,
            device,
            cam_img_num=cam_img_num,
            therm=therm,
            therm_levels=therm_levels,
        )

        CAMs_tab.append([CAMs_topo, CAMs_trivial])

        if img_idx == 0:
            test_out = test_model(
                model,
                test_loader,
                class_names,
                device,
                therm=therm,
                therm_levels=therm_levels,
                verbose=verbose,
            )

        if plots:
            # with plt.xkcd():
            if img_idx == 0:
                fig, ax = plt.subplots(1, 1, figsize=(16, 8))
                fig.suptitle(
                    f"Model : {model_name} | W = {np.unique(ds.W_tab)[0]}", fontsize=20
                )
                plt.tight_layout()
                ax.plot(ds.v_tab, ds.test_labels, lw=0, marker=6, label="True label")
                ax.plot(ds.v_tab, preds, lw=0, marker=7, label="Predicted label")
                ax.set_ylabel("Winding Number $\mathcal{W}$", fontsize=18)
                ax.set_xlabel("$\mathcal{v}$", fontsize=18)
                ax.set_yticks([0, 1])
                ax.axvline(x=1.0, ls="dotted")
                plt.legend(loc="lower left", fontsize=25)

            fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))

            CAMs_viz(
                fig_cam,
                ax_cam,
                [CAMs_topo, CAMs_trivial],
                ds,
                preds,
                class_names,
                cam_img_num=cam_img_num,
                M=50,
            )

    if test_dict:
        return test_out

    if cams_data_out:
        return CAMs_tab, preds


if __name__ == "__main__":
    # ds_clean = pathlib.Path("./Datasets/M=50/1000-200-100/")
    # ds_clean = pathlib.Path("./Datasets/M=50/2000-400-200/")
    ds_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    ds_w_001 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    ds_w_015 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    ds_w_100 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
    ds_w_300 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")

    therm_on = True
    ds_path = ds_clean
    now_beg = datetime.now()
    model_test(
        plots=False,
        sample_num=1,
        cams_data_out=False,
        therm=therm_on,
        therm_levels=100,
        model_arch=CNN_Upgrade_ThermEncod,
        model_name="upgraded_model_x_therm",
        test_path=ds_path,
        just_stats=True,
    )
    # model_test(plots=True, sample_num=1, cams_data_out=False,
    #         test_path=ds_path,
    #         model_name='/smart_models/smart_model_autogenerate_2',
    #         just_stats=False)
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint(" Total elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")
