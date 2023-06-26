#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 07:41:08 2022

@author: k4cp3rskiii
"""

import pathlib
import warnings
from datetime import datetime

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.metrics import classification_report, confusion_matrix
from termcolor import cprint

from CNN_dev import get_labels_comparison_fig
from CNN_testing import model_test
from Loaders import Importer
from Saliency import sal_test

warnings.filterwarnings("ignore")


# For memory profiling -> %load_ext memory_profiler -> %memit
# %load_ext snakeviz
# %snakeviz %mprun -f models_viz_datagen(samples_num=40, model_no=1)


def CAMs_Sal_Viz(
    fig,
    ax,
    CAMs_tab,
    sal_arr,
    ds_path,
    preds,
    class_names,
    model_name,
    cam_img_num=0,
    M=50,
):
    CAMs_topo, CAMs_trivial = CAMs_tab

    ds = Importer(ds_path, 100)

    test_loader = ds.get_test_loader()

    map_img = cm.coolwarm
    # map_img = cm.viridis

    map_hmap = cm.magma

    img = ds.test_data[cam_img_num]
    img_cmap = np.uint8(map_img(img) * 255)

    extent = 0, M, 0, M

    fig.suptitle(
        f"Model name : {model_name}\nDataset : {ds_path} | Img_num = {cam_img_num}\n Predicted label = {preds} | True "
        f"label = {test_loader.dataset.Y[cam_img_num]}\n {class_names[preds]} | "
        f"{class_names[test_loader.dataset.Y[cam_img_num]]}",
        size=25,
    )
    cbar2 = ax[0, 0].imshow(img_cmap, alpha=1, extent=extent, cmap=map_img)
    ax[0, 1].imshow(CAMs_topo[0], alpha=1, extent=extent, cmap=map_img)
    ax[0, 2].imshow(CAMs_trivial[0], alpha=1, extent=extent, cmap=map_img)
    ax[0, 0].set_title("Image", size=20)
    ax[0, 1].set_title("CAM - Topological", size=20)
    ax[0, 2].set_title("CAM - Trivial", size=20)

    ax[1, 0].set_title("Saliency", size=20)
    ax[1, 1].set_title("Img + CAM - Topological", size=20)
    ax[1, 2].set_title("Img + CAM - Trivial", size=20)
    cbar = ax[1, 0].imshow(sal_arr, alpha=1, extent=extent, cmap=map_hmap)
    ax[1, 1].imshow(img_cmap, alpha=0.65, extent=extent, cmap=map_img)
    ax[1, 1].imshow(CAMs_topo[0], alpha=0.35, extent=extent, cmap=map_img)
    ax[1, 2].imshow(img_cmap, alpha=0.65, extent=extent, cmap=map_img)
    ax[1, 2].imshow(CAMs_trivial[0], alpha=0.35, extent=extent, cmap=map_img)

    cax = plt.axes([1.02, 0.01, 0.075, 0.95])
    cax2 = plt.axes([-0.1, 0.01, 0.075, 0.95])
    plt.colorbar(cbar, cax=cax)
    plt.colorbar(cbar2, cax=cax2)

    fig.tight_layout()
    for axs in ax.flat:
        axs.grid(visible=False)
        axs.label_outer()

    plt.show()


def models_viz_datagen(
    model_type="smart",
    model_no=1,
    model_name_override="upgraded_model_x",
    ds_path=None,
    autogenerate=True,
    model_arch=None,
    samples_num=None,
    therm=False,
    therm_levels=100,
    verbose=True,
):
    if autogenerate:
        model_name = f"{model_type}_models/{model_type}_model_autogenerate_{model_no}"
    else:
        model_name = model_name_override

    if ds_path is None:
        # ds_path = pathlib.Path("./Datasets/M=50/1000-200-100/")
        # ds_path = pathlib.Path("./Datasets/M=50/2000-400-200/")
        ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=1-200-samples/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=3-200-samples/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=0.15-200-samples/")

    if verbose:
        cprint("[INFO]", "magenta", end=" ")
        cprint(f"Model name : {model_name}", end="\n")

        cprint("[INFO]", "magenta", end=" ")
        cprint("Test dataset path :", "white", end=" ")
        cprint(ds_path, "green", end="\n")

    sal_arr = sal_test(
        plots=False,
        test_path=ds_path,
        sal_return=True,
        sample_num=samples_num,
        model_name=model_name,
        therm=therm,
        therm_levels=therm_levels,
        model_arch=model_arch,
        verbose=verbose,
    )

    cam_arr, preds = model_test(
        plots=False,
        test_path=ds_path,
        sample_num=samples_num,
        cams_data_out=True,
        model_arch=model_arch,
        model_name=model_name,
        therm=therm,
        therm_levels=therm_levels,
        verbose=False,
    )

    return sal_arr, cam_arr, preds


def viz_dumb(viz_dict, ind, ds_ind, model_no, name_path_root):
    data_viz = viz_dict[ds_ind][model_no]

    ds_dict = {
        0: pathlib.Path("./Datasets/M=50/5000-1000-500/"),
        1: pathlib.Path("./Datasets/M=50/W=0.15-500-samples/"),
        2: pathlib.Path("./Datasets/M=50/W=1-500-samples/"),
        3: pathlib.Path("./Datasets/M=50/W=3-500-samples/"),
    }

    class_names = ["Trivial", "Topological"]

    fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))

    ind = ind

    sal_tab, CAMs_tab, pred_tab = data_viz

    ds_path = ds_dict[ds_ind]

    CAMs_Sal_Viz(
        fig_cam,
        ax_cam,
        CAMs_tab[ind],
        sal_tab[ind],
        ds_path,
        pred_tab[ind],
        class_names,
        cam_img_num=ind,
        model_name=f"{name_path_root}_{model_no}",
        M=50,
    )

    plt.show()

    fig_stat, ax_stat = plt.subplots(1, 1, figsize=(20, 10))

    ds = Importer(ds_path, 100)
    test_loader = ds.get_test_loader()

    plt.plot(
        ds.v_tab[ind],
        test_loader.dataset.Y[ind],
        marker="$|$",
        ls="",
        color="red",
        markersize=30,
    )

    get_labels_comparison_fig(fig_stat, ax_stat, ds=ds, preds=pred_tab)

    ax_stat.text(
        0.53,
        0.13,
        f"Classification Report                      \n\n{0}".format(
            classification_report(ds.test_labels, pred_tab, target_names=class_names)
        ),
        horizontalalignment="right",
        bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
        fontsize=16,
    )

    cmx = pd.DataFrame(
        confusion_matrix(ds.test_labels, pred_tab),
        index=["true:Topological", "true:        Trivial    "],
        columns=["pred:Topological", "pred:Trivial"],
    )

    ax_stat.text(
        -0.07,
        0.48,
        f"Confusion matrix\n\n" + cmx.to_string(),
        horizontalalignment="left",
        bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
        fontsize=16,
    )


def viz_smart(viz_dict, ind, ds_ind, model_no, name_path_root):
    data_viz = viz_dict[ds_ind][model_no]

    ds_dict = {
        0: pathlib.Path("./Datasets/M=50/5000-1000-500/"),
        1: pathlib.Path("./Datasets/M=50/W=0.15-500-samples/"),
        2: pathlib.Path("./Datasets/M=50/W=1-500-samples/"),
        3: pathlib.Path("./Datasets/M=50/W=3-500-samples/"),
    }

    class_names = ["Trivial", "Topological"]

    fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))

    ind = ind

    sal_tab, CAMs_tab, pred_tab = data_viz

    ds_path = ds_dict[ds_ind]

    CAMs_Sal_Viz(
        fig_cam,
        ax_cam,
        CAMs_tab[ind],
        sal_tab[ind],
        ds_path,
        pred_tab[ind],
        class_names,
        cam_img_num=ind,
        model_name=f"{name_path_root}_{model_no}",
        M=50,
    )

    plt.show()

    fig_stat, ax_stat = plt.subplots(1, 1, figsize=(20, 10))

    ds = Importer(ds_path, 100)
    test_loader = ds.get_test_loader()

    plt.plot(
        ds.v_tab[ind],
        test_loader.dataset.Y[ind],
        marker="$|$",
        ls="",
        color="red",
        markersize=30,
    )

    get_labels_comparison_fig(fig_stat, ax_stat, ds=ds, preds=pred_tab)

    ax_stat.text(
        0.53,
        0.13,
        f"Classification Report                      \n\n"
        + classification_report(
            ds.test_labels,
            pred_tab,
            target_names=class_names,
        ),
        horizontalalignment="right",
        bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
        fontsize=16,
    )

    cmx = pd.DataFrame(
        confusion_matrix(ds.test_labels, pred_tab),
        index=["true:Topological", "true:        Trivial    "],
        columns=["pred:Topological", "pred:Trivial"],
    )

    ax_stat.text(
        -0.07,
        0.48,
        f"Confusion matrix\n\n" + cmx.to_string(),
        horizontalalignment="left",
        bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
        fontsize=16,
    )


if __name__ == "__main__":
    for ds_path in [
        pathlib.Path("./Datasets/M=50/5000-1000-500/"),
        pathlib.Path("./Datasets/M=50/W=0.01-500-samples/"),
        pathlib.Path("./Datasets/M=50/W=0.05-500-samples/"),
        pathlib.Path("./Datasets/M=50/W=0.15-500-samples/"),
        pathlib.Path("./Datasets/M=50/W=1-500-samples/"),
        pathlib.Path("./Datasets/M=50/W=3-500-samples/"),
    ]:
        ind_max = 10
        ind_range = np.arange(0, ind_max, 1)

        now_beg = datetime.now()
        data_viz = models_viz_datagen(
            samples_num=ind_max, model_no=1, autogenerate=False, ds_path=ds_path
        )
        now = datetime.now()
        time_elapsed = now - now_beg
        elapsed_string = humanize.precisedelta(time_elapsed)

        cprint("[INFO]", "magenta", end=" ")
        cprint(" Total elapsed time =", end=" ")
        cprint(elapsed_string, "cyan", end="\n")

        class_names = ["Trivial", "Topological"]

        sal_tab, CAMs_tab, pred_tab = data_viz

        fig_stat, ax_stat = plt.subplots(1, 1, figsize=(20, 10))
        ds = Importer(ds_path, 100)
        test_loader = ds.get_test_loader()

        get_labels_comparison_fig(fig_stat, ax_stat, ds=ds, preds=pred_tab)

        ax_stat.text(
            0.53,
            0.13,
            f"Classification Report                      \n\n"
            + classification_report(ds.test_labels, pred_tab, target_names=class_names),
            horizontalalignment="right",
            bbox=dict(facecolor="Salmon", alpha=0.2),
            fontsize=16,
        )

        cmx = pd.DataFrame(
            confusion_matrix(ds.test_labels, pred_tab),
            index=["true:Topological", "true:        Trivial    "],
            columns=["pred:Topological", "pred:Trivial"],
        )

        ax_stat.text(
            -0.07,
            0.48,
            f"Confusion matrix\n\n" + cmx.to_string(),
            horizontalalignment="left",
            bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
            fontsize=16,
        )

        for ind in ind_range:
            fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))
            CAMs_Sal_Viz(
                fig_cam,
                ax_cam,
                CAMs_tab[ind],
                sal_tab[ind],
                ds_path,
                pred_tab[ind],
                class_names,
                cam_img_num=ind,
                model_name="upgraded_model_x",
                M=50,
            )
            plt.show()
            plt.close()
