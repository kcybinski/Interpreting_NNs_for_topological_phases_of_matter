import pathlib
from CNN_testing import model_test
from CNN_dev import get_preds_from_model
import numpy as np
from termcolor import cprint
from Models import CNN_Upgrade, CNN_Upgrade_ThermEncod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter


# %%


def print_class_report(dic, ret=False):
    report = (
        f"              precision    recall  f1-score   support\n\n"
        + f"    Trivial        {dic['Trivial']['precision']:.2f}      {dic['Trivial']['recall']:.2f}      {dic['Trivial']['f1-score']:.2f}     {dic['Trivial']['support']:.0f}\n"
        + f"Topological        {dic['Topological']['precision']:.2f}      {dic['Topological']['recall']:.2f}      {dic['Topological']['f1-score']:.2f}     {dic['Topological']['support']:.0f}\n\n"
        + f"accuracy                               {dic['accuracy']:.2f}     {dic['macro avg']['support']:.0f}\n"
        + f"macro avg          {dic['macro avg']['precision']:.2f}      {dic['macro avg']['recall']:.2f}      {dic['macro avg']['f1-score']:.2f}     {dic['macro avg']['support']:.0f}\n"
        + f"weighted avg       {dic['weighted avg']['precision']:.2f}      {dic['weighted avg']['recall']:.2f}      {dic['weighted avg']['f1-score']:.2f}     {dic['weighted avg']['support']:.0f}"
    )
    print(report)
    if ret:
        return report


def calc_general_stats(dic):
    test_results_dict = dic
    avg_stats = {
        "accuracy": np.array(
            [
                test_results_dict[w][real]["accuracy"]
                for w in test_results_dict.keys()
                for real in test_results_dict[w].keys()
            ]
        ).mean(),
        "Trivial": {
            "precision": np.array(
                [
                    test_results_dict[w][real]["Trivial"]["precision"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "recall": np.array(
                [
                    test_results_dict[w][real]["Trivial"]["recall"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "f1-score": np.array(
                [
                    test_results_dict[w][real]["Trivial"]["f1-score"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "support": np.array(
                [
                    test_results_dict[w][real]["Trivial"]["support"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).sum(),
        },
        "Topological": {
            "precision": np.array(
                [
                    test_results_dict[w][real]["Topological"]["precision"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "recall": np.array(
                [
                    test_results_dict[w][real]["Topological"]["recall"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "f1-score": np.array(
                [
                    test_results_dict[w][real]["Topological"]["f1-score"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "support": np.array(
                [
                    test_results_dict[w][real]["Topological"]["support"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).sum(),
        },
        "macro avg": {
            "precision": np.array(
                [
                    test_results_dict[w][real]["macro avg"]["precision"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "recall": np.array(
                [
                    test_results_dict[w][real]["macro avg"]["recall"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "f1-score": np.array(
                [
                    test_results_dict[w][real]["macro avg"]["f1-score"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "support": np.array(
                [
                    test_results_dict[w][real]["macro avg"]["support"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).sum(),
        },
        "weighted avg": {
            "precision": np.array(
                [
                    test_results_dict[w][real]["weighted avg"]["precision"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "recall": np.array(
                [
                    test_results_dict[w][real]["weighted avg"]["recall"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "f1-score": np.array(
                [
                    test_results_dict[w][real]["weighted avg"]["f1-score"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).mean(),
            "support": np.array(
                [
                    test_results_dict[w][real]["weighted avg"]["support"]
                    for w in test_results_dict.keys()
                    for real in test_results_dict[w].keys()
                ]
            ).sum(),
        },
    }
    return avg_stats


def test_generalization(
        models_tab = [1],
        n_realisations = 5,
        n_slices = 40,
        w_low = 1.0,
        w_high = 4.9,
        v_ticks_num = 11,
        verbose = False,
        save = False,
        show = True,
        model_arch = None,
        model_type = "dumb",
        folder = None,
        name_override="upgraded_model_x",
        therm = False,
        levels = 10,

        ):
    
    for model_num in models_tab:
        w_lab_high = np.floor(w_high) if w_high < np.floor(w_high)+0.5 else np.floor(w_high)+0.5
        w_values = np.linspace(w_low, w_high, n_slices, endpoint=True)
        w_labels = np.linspace(w_low, w_lab_high, 8, endpoint=True)
        v_tab = np.linspace(0, 2, v_ticks_num, endpoint=True)
        test_results_dict = {}
        test_preds_dict = {}
        test_targets_dict = {}
        avg_preds_mat = []
        avg_targets_mat = []
        kind = "therm_" if therm else ""
        if folder is not None:
            model_name = (
                f"{folder}/{model_type}_models/{model_type}_model_{kind}autogenerate_{model_num}"
            )
        else:
            model_name=name_override
        cprint("[INFO]", "magenta", end=" ")
        cprint("Testing model: ", "green", end=" ")
        cprint(model_name, "cyan", end="\n")
        for W in w_values:
            test_results_dict[f"W={W:.1f}"] = {}
            test_preds_dict[f"W={W:.1f}"] = {}
            test_targets_dict[f"W={W:.1f}"] = {}
            p = pathlib.Path(
                f"./Datasets/M=50/{n_realisations}-disorder-realisations/W={W:.1f}/"
            )
            if verbose:
                cprint("[INFO]", "magenta", end=" ")
                cprint("Test dataset path :", "white", end=" ")
                cprint(p, "green", end="\n")
            for real in range(n_realisations):
                if verbose:
                    cprint("[INFO]", "magenta", end=" ")
                    cprint("Realisation :", "white", end=" ")
                    cprint(real, "green", end="\n")
                test_dict, preds, targets = model_test(
                    plots=False,
                    test_dict=True,
                    test_path=p,
                    model_name=model_name,
                    model_arch=model_arch,
                    just_stats_and_preds=True,
                    verbose=verbose,
                    realization=real,
                    therm=therm,
                    therm_levels=levels,
                )
                test_results_dict[f"W={W:.1f}"][f"real={real}"] = test_dict
                test_preds_dict[f"W={W:.1f}"][f"real={real}"] = preds
                test_targets_dict[f"W={W:.1f}"][f"real={real}"] = targets

                # %%

            avg_p = np.array(
                [
                    np.array(test_preds_dict[f"W={W:.1f}"][f"real={r}"])
                    for r in range(n_realisations)
                ]
            ).mean(axis=0)
            avg_t = np.array(
                [
                    np.array(test_targets_dict[f"W={W:.1f}"][f"real={r}"])
                    for r in range(n_realisations)
                ]
            ).mean(axis=0)
            avg_preds_mat.append(avg_p)
            avg_targets_mat.append(avg_t)

        # %%
        avg_stats = calc_general_stats(test_results_dict)
        cprint("[INFO]", "magenta", end=" ")
        cprint("Total classification report", "green", end="\n")
        cprint("[INFO]", "magenta", end=" ")
        cprint("Model: ", "green", end=" ")
        cprint(model_name, "cyan", end="\n")

        fontsize = 60
        ticksize = 32

        sns.set(font_scale=2.5)

        avg_preds_mat = np.array(avg_preds_mat)
        fig, axs = plt.subplots(1, 3, figsize=(35, 10))
        ax = axs[0]
        image = gaussian_filter(avg_preds_mat[:, ::-1].T, sigma=1.5)
        sns.heatmap(image, ax=ax, cmap="coolwarm", rasterized=True, vmin=0, vmax=1)
        ymin, ymax = ax.get_ylim()
        ytick_pos = np.linspace(ymin, ymax, v_ticks_num)
        xmin, xmax = ax.get_xlim()
        xtick_pos = np.linspace(xmin, xmax, len(w_labels))
        ax.set_title("Predictions", size=fontsize)
        ax.set_xticks(xtick_pos)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([f"{t:.1f}" for t in v_tab], size=ticksize)
        ax.set_xticklabels([f"{t:.1f}" for t in w_labels], size=ticksize)
        ax.set_xlabel("W", size=fontsize)
        ax.set_ylabel("v", size=fontsize)
        # plt.show()

        avg_targets_mat = np.array(avg_targets_mat)
        ax = axs[1]
        image = gaussian_filter(avg_targets_mat[:, ::-1].T, sigma=1.5)
        sns.heatmap(image, ax=ax, cmap="coolwarm", rasterized=True, vmin=0, vmax=1)
        ymin, ymax = ax.get_ylim()
        ytick_pos = np.linspace(ymin, ymax, v_ticks_num)
        xmin, xmax = ax.get_xlim()
        xtick_pos = np.linspace(xmin, xmax, len(w_labels))
        ax.set_title("Targets", size=fontsize)
        ax.set_xticks(xtick_pos)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels([f"{t:.1f}" for t in v_tab], size=ticksize)
        ax.set_xticklabels([f"{t:.1f}" for t in w_labels], size=ticksize)
        ax.set_xlabel("W", size=fontsize)
        ax.set_ylabel("v", size=fontsize)
        # plt.show()

        ax = axs[2]
        ax.text(
            0.70,
            0.85,
            "Total classification report",
            horizontalalignment="right",
            fontsize=25,
        )
        ax.text(
            0.83,
            0.35,
            print_class_report(dic=avg_stats, ret=True),
            horizontalalignment="right",
            bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
            fontsize=20,
        )
        ax.axis("off")

        plt.suptitle(f"Model : {model_name}", size=30)
        plt.tight_layout(pad=2)
        if save:
            fig.savefig(
                f"Models/Model_generalizations/{folder}_{model_type}_model_autogenerate_{model_num}.pdf",
                dpi=300,
            )
        if not save:
            if show:
                plt.show()
        plt.close()


if __name__ == "__main__":
    # Comment-out the line above, and uncomment one below for automated generation of images
# for model_num in [4, 6, 12, 13, 15, 22, 25, 26, 28, 29]:
    n_realisations = 5
    n_slices = 40
    w_values = np.linspace(1, 4.9, n_slices, endpoint=True)
    w_labels = np.linspace(1, 4.5, 8, endpoint=True)
    v_ticks_num = 11
    v_tab = np.linspace(0, 2, v_ticks_num, endpoint=True)
    test_results_dict = {}
    test_preds_dict = {}
    test_targets_dict = {}
    avg_preds_mat = []
    avg_targets_mat = []
    # Toggles printing of statistics on each realization
    verbose = False
    # Toggles image save
    save = False
    # Toggles image showing if it is not getting saved
    show = True
    """
    Model_arch:
    * CNN_Upgrade - None
    * CNN_Upgrade_ThermEncod - CNN_Upgrade_ThermEncod
    """
    model_arch = CNN_Upgrade_ThermEncod
    model_type = "dumb"
    # model_num = 10
    folder = "therm_10_lev"
    therm = True
    levels = 10
    kind = "therm_" if therm else ""
    # Models (autogen_3): 4, 6(?), 12(?), 13, 15, 22, 25(?), 26, 28, 29(?)
    # Notatka: Porównać 36 z resztą (wyjątek?), 46, 48 (false positive)
    model_name = (
        f"{folder}/{model_type}_models/{model_type}_model_{kind}autogenerate_{model_num}"
    )
    cprint("[INFO]", "magenta", end=" ")
    cprint("Testing model: ", "green", end=" ")
    cprint(model_name, "cyan", end="\n")
    for W in w_values:
        test_results_dict[f"W={W:.1f}"] = {}
        test_preds_dict[f"W={W:.1f}"] = {}
        test_targets_dict[f"W={W:.1f}"] = {}
        p = pathlib.Path(
            f"./Datasets/M=50/{n_realisations}-disorder-realisations/W={W:.1f}/"
        )
        if verbose:
            cprint("[INFO]", "magenta", end=" ")
            cprint("Test dataset path :", "white", end=" ")
            cprint(p, "green", end="\n")
        for real in range(n_realisations):
            if verbose:
                cprint("[INFO]", "magenta", end=" ")
                cprint("Realisation :", "white", end=" ")
                cprint(real, "green", end="\n")
            test_dict, preds, targets = model_test(
                plots=False,
                test_dict=True,
                test_path=p,
                model_name=model_name,
                model_arch=model_arch,
                just_stats_and_preds=True,
                verbose=verbose,
                realization=real,
                therm=therm,
                therm_levels=levels,
            )
            test_results_dict[f"W={W:.1f}"][f"real={real}"] = test_dict
            test_preds_dict[f"W={W:.1f}"][f"real={real}"] = preds
            test_targets_dict[f"W={W:.1f}"][f"real={real}"] = targets

            # %%

        avg_p = np.array(
            [
                np.array(test_preds_dict[f"W={W:.1f}"][f"real={r}"])
                for r in range(n_realisations)
            ]
        ).mean(axis=0)
        avg_t = np.array(
            [
                np.array(test_targets_dict[f"W={W:.1f}"][f"real={r}"])
                for r in range(n_realisations)
            ]
        ).mean(axis=0)
        avg_preds_mat.append(avg_p)
        avg_targets_mat.append(avg_t)

    # %%
    avg_stats = calc_general_stats(test_results_dict)
    cprint("[INFO]", "magenta", end=" ")
    cprint("Total classification report", "green", end="\n")
    cprint("[INFO]", "magenta", end=" ")
    cprint("Model: ", "green", end=" ")
    cprint(model_name, "cyan", end="\n")

    fontsize = 60
    ticksize = 32

    sns.set(font_scale=2.5)

    avg_preds_mat = np.array(avg_preds_mat)
    fig, axs = plt.subplots(1, 3, figsize=(35, 10))
    ax = axs[0]
    image = gaussian_filter(avg_preds_mat[:, ::-1].T, sigma=1.5)
    sns.heatmap(image, ax=ax, cmap="coolwarm", rasterized=True, vmin=0, vmax=1)
    ymin, ymax = ax.get_ylim()
    ytick_pos = np.linspace(ymin, ymax, v_ticks_num)
    xmin, xmax = ax.get_xlim()
    xtick_pos = np.linspace(xmin, xmax, len(w_labels))
    ax.set_title("Predictions", size=fontsize)
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{t:.1f}" for t in v_tab], size=ticksize)
    ax.set_xticklabels([f"{t:.1f}" for t in w_labels], size=ticksize)
    ax.set_xlabel("W", size=fontsize)
    ax.set_ylabel("v", size=fontsize)
    # plt.show()

    avg_targets_mat = np.array(avg_targets_mat)
    ax = axs[1]
    image = gaussian_filter(avg_targets_mat[:, ::-1].T, sigma=1.5)
    sns.heatmap(image, ax=ax, cmap="coolwarm", rasterized=True, vmin=0, vmax=1)
    ymin, ymax = ax.get_ylim()
    ytick_pos = np.linspace(ymin, ymax, v_ticks_num)
    xmin, xmax = ax.get_xlim()
    xtick_pos = np.linspace(xmin, xmax, len(w_labels))
    ax.set_title("Targets", size=fontsize)
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"{t:.1f}" for t in v_tab], size=ticksize)
    ax.set_xticklabels([f"{t:.1f}" for t in w_labels], size=ticksize)
    ax.set_xlabel("W", size=fontsize)
    ax.set_ylabel("v", size=fontsize)
    # plt.show()

    ax = axs[2]
    ax.text(
        0.70,
        0.85,
        "Total classification report",
        horizontalalignment="right",
        fontsize=25,
    )
    ax.text(
        0.83,
        0.35,
        print_class_report(dic=avg_stats, ret=True),
        horizontalalignment="right",
        bbox=dict(facecolor="Salmon", alpha=0.2, boxstyle="round"),
        fontsize=20,
    )
    ax.axis("off")

    plt.suptitle(f"Model : {model_name}", size=30)
    plt.tight_layout(pad=2)
    if save:
        fig.savefig(
            f"Models/Model_generalizations/{folder}_{model_type}_model_autogenerate_{model_num}.pdf",
            dpi=300,
        )
    if not save:
        if show:
            plt.show()
    plt.close()
