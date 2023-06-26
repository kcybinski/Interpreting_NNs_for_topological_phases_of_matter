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



if __name__ == "__main__":
    # Begin time
    now_beg = datetime.now()

    num = 0
    smart_num = 0
    dumb_num = 0
    it_num = 1
    dumb_count = 10
    smart_count = 10
    model_num = dumb_count + smart_count
    while num <= model_num:
        date_string = now_beg.strftime("%d/%m/%Y")
        time_string = now_beg.strftime("%H:%M:%S")
        cprint("[INFO]", "magenta", end=" ")
        cprint("Beginning time = ", end=" ")
        cprint(date_string, "white", end=" ")
        cprint(time_string, "cyan", end="\n")

        # End time
        now = datetime.now()
        date_string = now.strftime("%d/%m/%Y")
        time_string = now.strftime("%H:%M:%S")
        cprint("[INFO]", "magenta", end=" ")
        cprint(" Current  time = ", end=" ")
        cprint(date_string, "white", end=" ")
        cprint(time_string, "cyan", end="\n")

        time_elapsed = now - now_beg
        # s = time_elapsed.seconds
        # hours, remainder = divmod(s, 3600)
        # minutes, seconds = divmod(remainder, 60)
        # elapsed_string = '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
        elapsed_string = humanize.precisedelta(time_elapsed)

        cprint("[INFO]", "magenta", end=" ")
        cprint(" Total elapsed time =", end=" ")
        cprint(elapsed_string, "cyan", end="\n")

        cprint("[INFO]", "magenta", end=" ")
        cprint(" Current  iteration =", end=" ")
        cprint(it_num, "white", end="\n")
        

        cprint("[INFO]", "magenta", end=" ")
        cprint(" Avg time/iteration =", end=" ")
        cprint(humanize.precisedelta(time_elapsed.seconds/it_num), "white", end="\n")
        it_num += 1


        # Model training
        if sys.platform.startswith("darwin"):
            os.system("python3 CNN_dev.py")
        else:
            os.system("python CNN_dev.py")


        test_basic_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")
        test_basic_path_2 = pathlib.Path("./Datasets/M=50/W=0.01-500-samples/")
        test_basic_path_3 = pathlib.Path("./Datasets/M=50/W=0.05-500-samples/")
        test_basic_path_4 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
        cprint("[INFO]", "magenta", end=" ")
        cprint("Test dataset path :", 'white', end=" ")
        cprint(test_basic_path, "green", end="\n")
        test_dict_basic = model_test(plots=False, test_dict=True,
                                     test_path=test_basic_path,
                                     just_stats=True,
                                     )
        cprint("[INFO]", "magenta", end=" ")
        cprint("Test dataset path :", 'white', end=" ")
        cprint(test_basic_path_2, "green", end="\n")
        test_dict_basic_2 = model_test(plots=False, test_dict=True,
                                       test_path=test_basic_path_2,
                                       just_stats=True,
                                       )
        cprint("[INFO]", "magenta", end=" ")
        cprint("Test dataset path :", 'white', end=" ")
        cprint(test_basic_path_3, "green", end="\n")
        test_dict_basic_3 = model_test(plots=False, test_dict=True,
                                       test_path=test_basic_path_3,
                                       just_stats=True,
                                       )
        cprint("[INFO]", "magenta", end=" ")
        cprint("Test dataset path :", 'white', end=" ")
        cprint(test_basic_path_4, "green", end="\n")
        test_dict_basic_4 = model_test(plots=False, test_dict=True,
                                       test_path=test_basic_path_4,
                                       just_stats=True,
                                       )

        if (
            test_dict_basic["macro avg"]["precision"] >= 0.95
            and test_dict_basic["macro avg"]["recall"] >= 0.95
            and test_dict_basic_2["macro avg"]["precision"] >= 0.95
            and test_dict_basic_2["macro avg"]["recall"] >= 0.95
            and test_dict_basic_3["macro avg"]["precision"] >= 0.95
            and test_dict_basic_3["macro avg"]["recall"] >= 0.95
            and test_dict_basic_4["macro avg"]["precision"] >= 0.95
            and test_dict_basic_4["macro avg"]["recall"] >= 0.95
        ):
            # test_disord_path = pathlib.Path("./Datasets/M=50/W=1-200-samples/")
            test_disord_path = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
            cprint("[INFO]", "magenta", end=" ")
            cprint("Test dataset path :", 'white', end=" ")
            cprint(test_disord_path, "green", end="\n")
            test_dict_disord = model_test(plots=False, test_dict=True,
                                          test_path=test_disord_path,
                                          just_stats=True,
                                          )
            # if (
            #     test_dict_disord["macro avg"]["precision"] > 0.9
            #     and test_dict_disord["macro avg"]["recall"] > 0.9
            # ):
            if (
                test_dict_disord["macro avg"]["precision"] > 0.65
                and test_dict_disord["macro avg"]["recall"] > 0.65
            ):
                smart_num += 1
                if smart_num <= smart_count:
                    num += 1
                # Model renaming
                old_name = "Models/upgraded_model_x.dict"
                new_name = f"Models/smart_Models/smart_model_autogenerate_{smart_num}.dict"
                os.rename(old_name, new_name)

                # History file renaming
                old_name_hist = "Models/upgraded_model_x_history.pickle"
                new_name_hist = f"Models/smart_Models/smart_model_autogenerate_{smart_num}_history.pickle"
                os.rename(old_name_hist, new_name_hist)

            else:
                dumb_num += 1
                if dumb_num <= dumb_count:
                    num += 1
                # Model renaming
                old_name = "Models/upgraded_model_x.dict"
                new_name = f"Models/dumb_Models/dumb_model_autogenerate_{dumb_num}.dict"
                os.rename(old_name, new_name)

                # History file renaming
                old_name_hist = "Models/upgraded_model_x_history.pickle"
                new_name_hist = f"Models/dumb_Models/dumb_model_autogenerate_{dumb_num}_history.pickle"
                os.rename(old_name_hist, new_name_hist)

        cprint("\nFound models :", "cyan", end=" ")
        if num == 0:
            cprint(f"{num}/{model_num}", "red")
        else:
            cprint(f"{num}/{model_num}", "green", attrs=["bold", "blink"])

        cprint("\nFound smart models :", "cyan", end=" ")
        if smart_num == 0:
            cprint(f"{smart_num}/{smart_count}", "red")
        else:
            cprint(f"{smart_num}/{smart_count}", "green", attrs=["bold", "blink"])


        cprint("\nFound smart-dumb models :", "cyan", end=" ")
        if dumb_num == 0:
            cprint(f"{dumb_num}/{dumb_count}", "red")
        else:
            cprint(f"{dumb_num}/{dumb_count}", "green", attrs=["bold", "blink"])

        cprint("\n==== NEXT RUN ====\n", "cyan")
