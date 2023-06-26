import pathlib
from datetime import datetime

import humanize
import matplotlib.pyplot as plt
from termcolor import cprint
import pickle

from CAM_Sal_viz_utils import models_viz_datagen, CAMs_Sal_Viz
from Models import CNN_Upgrade_ThermEncod
from tqdm import trange, tqdm

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
    parent_folder = "autosearch_therm_1_model"
    model_type = "dumb"
    model_no = 1
    fname = f'{parent_folder}/{model_type}_models/{model_type}_model_therm_autogenerate_{model_no}'
    ds_viz = ds_clean
    dumb_count = 1
    smart_count = 0
    datasets = [ds_clean, ds_w_015, ds_w_100, ds_w_300]
    gen_date = "1_03_2023"

    cprint("[INFO]", "magenta", end=" ")
    cprint(" Generating visualizations for folder:", end=" ")
    cprint(f" {parent_folder}", "cyan")

    # %%

    for model_type, count in tqdm([("dumb", dumb_count), ("smart", smart_count)], colour="blue", unit="type"):
        viz_all = {}
        for ds_num, ds_viz in enumerate(tqdm(datasets, colour="green", unit="ds")):
            for model_no in trange(1, count+1, colour="cyan", unit="model"):
                # If 'None', then all dataset is processed
                samples = None

                now_beg = datetime.now()

                data_viz = models_viz_datagen(samples_num=samples,
                                              autogenerate=False,
                                              model_name_override=fname,
                                              model_arch=CNN_Upgrade_ThermEncod,
                                              ds_path=ds_viz,
                                              therm=therm_on,
                                              therm_levels=levels,
                                              verbose=False)
                viz_all[ds_num] = data_viz
                now = datetime.now()
                time_elapsed = now - now_beg
                elapsed_string = humanize.precisedelta(time_elapsed)

                cprint("[INFO]", "magenta", end=" ")
                cprint(" Total elapsed time =", end=" ")
                cprint(elapsed_string, "cyan", end="\n")

        with open(f"Models/viz_all_{model_type}_therm_{gen_date}.pickle", "wb") as f:
            pickle.dump(viz_all, f)
