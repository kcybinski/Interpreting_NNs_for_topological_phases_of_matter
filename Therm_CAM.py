import pathlib
from datetime import datetime

import humanize
import matplotlib.pyplot as plt
from termcolor import cprint

from CAM_Sal_viz_utils import models_viz_datagen, CAMs_Sal_Viz
from Models import CNN_Upgrade_ThermEncod

if __name__ == "__main__":
    # ds_clean = pathlib.Path("./Datasets/M=50/1000-200-100/")
    # ds_clean = pathlib.Path("./Datasets/M=50/2000-400-200/")
    ds_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
    ds_w_001 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
    ds_w_005 = pathlib.Path("./Datasets/M=50/W=0.05-500-samples/")
    ds_w_015 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
    ds_w_100 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
    ds_w_300 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")

    # Thermometer encoding levels
    therm_on = True
    levels = 100
    fname = 'autosearch_therm_1_model/dumb_models/dumb_model_therm_autogenerate_1'
    ds_viz = ds_clean

    # %%

    # If 'None', then all dataset is processed
    samples = 5

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

    # %%
    for ind in range(samples):
        fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10))
        CAMs_Sal_Viz(fig_cam, ax_cam, CAMs_tab[ind], sal_tab[ind], ds_viz,
                     pred_tab[ind], class_names, cam_img_num=ind,
                     model_name=f"{fname}", M=50)
