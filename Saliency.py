#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 01:10:37 2022

@author: k4cp3rskiii
"""

import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from matplotlib import cm

# Pytorch imports
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.transforms import ToTensor

from torchvision import transforms
import pathlib

from Loaders import Importer
from Models import CNN_Upgrade
from CNN_dev import get_preds_from_model, test_model, returnCAM, get_saliency_map


def sal_test(
    plots=True,
    test_dict=False,
    test_path=None,
    sal_return=False,
    model_name=None,
    sample_num=None,
    M=50,
    therm=False,
    therm_levels=100,
    model_arch=None,
    verbose=True,
):
    """
    Function for generating saliency map of test dataset predictions

    Parameters
    ----------
    plots : TYPE, optional
        DESCRIPTION. The default is True.
    test_dict : TYPE, optional
        DESCRIPTION. The default is False.
    test_path : TYPE, optional
        DESCRIPTION. The default is None.
    sal_return : TYPE, optional
        DESCRIPTION. The default is False.
    model_name : TYPE, optional
        DESCRIPTION. The default is None.
    sample_num : TYPE, optional
        DESCRIPTION. The default is None.
    M : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    None.

    """

    # =============================================================================
    #     device = torch.device(
    #         "cuda:0"
    #         if torch.cuda.is_available()
    #         else "mps"
    #         if torch.backends.mps.is_available()
    #         else "cpu"
    #     )
    # =============================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if test_path is None:
        # ds_path = pathlib.Path("./Datasets/M=50/1000-200-100/")
        # ds_path = pathlib.Path("./Datasets/M=50/2000-400-200/")
        ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=1-200-samples/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=3-200-samples/")
        # ds_path = pathlib.Path("./Datasets/M=50/W=0.15-200-samples/")
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

    # Setting levels to `none` leaves the base cases unchanged.
    # But if not none, it encodes the therm_encoding into the test loader
    levels = None
    if therm:
        levels = therm_levels
    ds = Importer(ds_path, def_batch_size, therm_levels=levels)

    test_loader = ds.get_test_loader()
    # model = torch.load(f'Models/{model_name}.pt').float().to(device)

    CAMs_tab = []

    if sample_num is None:
        sample_num = test_loader.dataset.X.shape[0]
    for img_idx in range(sample_num):

        if model_arch is None:
            model = CNN_Upgrade()
        else:
            model = model_arch(therm_levels)
        model.load_state_dict(
            torch.load(f"Models/{model_name}.dict", map_location=torch.device("cpu"))
        )
        model = model.float().to(device)

        cam_img_num = img_idx

        sal_input = test_loader.dataset.X

        sal_arr = get_saliency_map(sal_input, model, M, device)

        preds = get_preds_from_model(
            model,
            test_loader,
            class_names,
            device,
            therm=therm,
            therm_levels=therm_levels,
        )

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
        # %%
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

            # %%

            features_blobs = []

            def hook_feature(module, input, output):
                features_blobs.append(output.data.cpu().numpy())

            model._modules.get("cnn_layers_2").register_forward_hook(hook_feature)

            # Tutaj wyciągamy wagi, które pojawiają się podczas klasyfikacji, przy softmaxie (po stronie GAPa,
            # stąd 'params[-2]', czyli druga od końca)
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
            # print(weight_softmax)

            # %%

            transform = transforms.Compose([ToTensor()])
            img_tensor = (
                transform(ds.test_data[cam_img_num])
                .float()
                .reshape(-1, 1, 50, 50)
                .to(device)
            )
            img_variable = Variable(img_tensor)
            # # Teraz predykcja! Hak przyczepiony do ostatniej warstwy konwolucyjnej
            # # zapamięta do czego skolapsował obrazek.
            logit = model(img_variable)
            logit

            # %%

            # Puszczamy przez softmax:
            h_x = F.softmax(logit, dim=1).data.squeeze()

            # Sortujemy prawdopodobieństwa wraz z odpowiadającymi im kategoriami
            probs, idx = h_x.cpu().sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()

            # Wreszcie możemy wygenerować CAM dla dowolnej kategorii. Zacznijmy od klasy z największym
            # prawdopodobieństwem, czyli top1: idx[0] (bo sortowaliśmy względem prawdopodobieństw, pamiętacie?)
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

            # %%

            map_img = cm.coolwarm

            map_hmap = cm.magma

            img = ds.test_data[cam_img_num]
            img_cmap = np.uint8(map_img(img) * 255)

            extent = 0, 50, 0, 50

            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            # fig.suptitle(f"Img_num = {cam_img_num} | dataset = {ds_path} \n model = {model_name}", size=20, y=0.8)
            cbar2 = ax[0, 0].imshow(img_cmap, alpha=1, extent=extent, cmap=map_img)
            cbar = ax[0, 1].imshow(CAMs[0], alpha=1, extent=extent, cmap=map_hmap)
            ax[1, 0].imshow(sal_arr[img_idx], alpha=1, extent=extent, cmap=map_hmap)
            ax[1, 1].imshow(
                sal_arr[img_idx],
                alpha=0.9,
                extent=extent,
                cmap=map_hmap,
                interpolation="gaussian",
            )
            ax[1, 1].imshow(img_cmap, alpha=0.5, extent=extent, cmap=map_img)
            ax[0, 0].set_title("Image")
            ax[0, 1].set_title("CAM")
            ax[1, 0].set_title("Saliency")
            ax[1, 1].set_title("Saliency+Img")
            # cax = plt.axes([0.95, 0.25, 0.075, 0.5])
            # cax2 = plt.axes([-0.02, 0.25, 0.075, 0.5])

            cax = plt.axes([0.95, 0.30, 0.075, 0.4])
            cax2 = plt.axes([-0.02, 0.30, 0.075, 0.4])
            plt.colorbar(cbar, cax=cax)
            plt.colorbar(cbar2, cax=cax2)
            # plt.axis('off')

            for axs in ax.flat:
                axs.grid(visible=False)
                axs.label_outer()

            # plt.savefig("topological_CAM_better.pdf", bbox_inches='tight')
            plt.show()

        if test_dict:
            return test_out
        elif sal_return:
            return sal_arr


if __name__ == "__main__":
    # sal_test(plots=True, sal_return=False)
    sal_test(
        plots=True,
        sal_return=False,
        sample_num=5,
        test_path=pathlib.Path("./Datasets/M=50/W=0.15-500-samples/"),
    )

    sal_test(
        plots=True,
        sal_return=False,
        sample_num=5,
        test_path=pathlib.Path("./Datasets/M=50/W=1-500-samples/"),
    )

    sal_test(
        plots=True,
        sal_return=False,
        sample_num=5,
        test_path=pathlib.Path("./Datasets/M=50/W=3-500-samples/"),
    )
