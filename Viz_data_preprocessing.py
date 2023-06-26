#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:38:51 2022

@author: k4cp3rskiii
"""

# %% [markdown]
# # CNN Development

# %% [markdown]
# ## Initial setup

# %% [markdown]
# ### Import zone

# %% [markdown]
# Running this cell for the first time may take several seconds, since the physics backend is coded with JIT-compilable functions for greater performance, and their compilation takes place during import.

# %%
import pickle

from termcolor import cprint
import pathlib
from datetime import datetime
import humanize

import warnings
warnings.filterwarnings("ignore")


# Pytorch imports
import torch

import torch.optim as optim

from Loaders import Importer
from Models import CNN_Upgrade
from CAM_Sal_viz_utils import models_viz_datagen

# %% [markdown]
# ### Dataset downloading

# %% [markdown]
# 

# Datasets used here are too powerful to be distributed all in git. Therefore they are shared separately.
# This is how to obtain all the necessary dataset files, along with folder structure.

# Files can be found at [this link](https://drive.google.com/drive/folders/1-13Wwsi7yukRb3QafZ5i7_CfjT8hTBjR?usp=share_link)

# Structure:

# ```
# 📦Datasets
#  ┣ 📂M=50
#  ┃ ┣ 📂1000-200-100
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂2000-400-200
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂5000-1000-500
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂W=0.15-200-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┣ 📂W=0.15-500-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┣ 📂W=1-200-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┣ 📂W=1-500-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┣ 📂W=3-200-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┗ 📂W=3-500-samples
#  ┃   ┗ 📜test_set.pickle
#  ┣ 📂M=80
#  ┃ ┣ 📂1000-200-100
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂2000-400-200
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂5000-1000-500
#  ┃ ┃ ┣ 📜test_set.pickle
#  ┃ ┃ ┣ 📜training_set.pickle
#  ┃ ┃ ┗ 📜validation_set.pickle
#  ┃ ┣ 📂W=1-500-samples
#  ┃ ┃ ┗ 📜test_set.pickle
#  ┃ ┗ 📂W=3-500-samples
#  ┃   ┗ 📜test_set.pickle
#  ┗ 📜readme.md
#  ```

# %%
def_batch_size = 250
epochs = 100

lr = 0.001
moment = 0.99
# ==============================================================================
# Here we only have two classes, defined by Winding number:
# Winding number = 0
# Winding number = 1
# ==============================================================================
num_classes = 2 
class_names = ["Trivial", "Topological"]

device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")
ds_path = pathlib.Path("./Datasets/M=50/5000-1000-500/")


ds = Importer(ds_path, def_batch_size)
train_loader = ds.get_train_loader(seed=2137)
val_loader = ds.get_val_loader()
test_loader = ds.get_test_loader()

model  = CNN_Upgrade()
model_load = False
if model_load:
    model_name = 'upgraded_model_x'
    model.load_state_dict(torch.load(f"Models/{model_name}.dict"))
    # with open(f"./Models/{model_name}_history.pickle", "rb") as f:
    #     train_history = pickle.load(f)

model = model.float()
model.to(device);
optimizer = optim.SGD(model.parameters(), 
                      lr=lr, 
                      momentum=moment, 
                      weight_decay=0.1)


# %% [markdown]
# ## Models visualization
# Dated : 23.11.2022

gen_date = '23_11_2022'

# %% [markdown]
# ### Dumb models data generation

# %% [markdown]
# #### Clean data

# %%
dumb_count = 28

# %%
data_viz_clean_dumb = {}
path_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    model_name = f"threshold_at_w_1/dumb_models/dumb_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_clean, model_name_override=model_name, autogenerate=False)
    data_viz_clean_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 0.15

# %%
data_viz_w15_dumb = {}
path_w15 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    model_name = f"threshold_at_w_1/dumb_models/dumb_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w15, model_name_override=model_name, autogenerate=False)
    data_viz_w15_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 1

# %%
data_viz_w1_dumb = {}
path_w1 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    model_name = f"threshold_at_w_1/dumb_models/dumb_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w1, model_name_override=model_name, autogenerate=False)
    data_viz_w1_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 3

# %%
data_viz_w3_dumb = {}
path_w3 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    model_name = f"threshold_at_w_1/dumb_models/dumb_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w3, model_name_override=model_name, autogenerate=False)
    data_viz_w3_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %%
viz_all_dumb = {0: data_viz_clean_dumb, 1: data_viz_w15_dumb, 2: data_viz_w1_dumb, 3: data_viz_w3_dumb}

# %%
with open(f"Models/viz_all_dumb_{gen_date}.pickle", "wb") as f:
    pickle.dump(viz_all_dumb, f)

# %% [markdown]
# ### Smart models data generation

# %% [markdown]
# #### Clean data

# %%
smart_count = 8

# %%
data_viz_clean_smart = {}
path_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    model_name = f"threshold_at_w_1/smart_models/smart_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_clean, model_name_override=model_name, autogenerate=False)
    data_viz_clean_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 0.15

# %%
data_viz_w15_smart = {}
path_w15 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    model_name = f"threshold_at_w_1/smart_models/smart_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w15, model_name_override=model_name, autogenerate=False)
    data_viz_w15_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 1

# %%
data_viz_w1_smart = {}
path_w1 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    model_name = f"threshold_at_w_1/smart_models/smart_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w1, model_name_override=model_name, autogenerate=False)
    data_viz_w1_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 3

# %%
data_viz_w3_smart = {}
path_w3 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    model_name = f"threshold_at_w_1/smart_models/smart_model_autogenerate_{model_ind}"
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w3, model_name_override=model_name, autogenerate=False)
    data_viz_w3_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %%
viz_all_smart = {0: data_viz_clean_smart, 1: data_viz_w15_smart, 2: data_viz_w1_smart, 3: data_viz_w3_smart}

# %%
with open(f"Models/viz_all_smart_{gen_date}.pickle", "wb") as f:
    pickle.dump(viz_all_smart, f)

# %% [markdown]
# ## Models visualization
# Dated : 28.11.2022

# %% [markdown]
# ### Dumb models data generation

# %%
gen_date = '28_11_2022'

# %% [markdown]
# #### Clean data

# %%
dumb_count = 316

# %%
data_viz_clean_dumb = {}
path_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_clean)
    data_viz_clean_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    it_time = time_elapsed
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint("Dataset elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(it_time*(dumb_count-model_ind))

    cprint("[INFO]", "magenta", end=" ")
    cprint("Expected time left =", end=" ")
    cprint(elapsed_string, "cyan", end="\n\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 0.15

# %%
data_viz_w15_dumb = {}
path_w15 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w15)
    data_viz_w15_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
        
    it_time = time_elapsed
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint("Dataset elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(it_time*(dumb_count-model_ind))

    cprint("[INFO]", "magenta", end=" ")
    cprint("Expected time left =", end=" ")
    cprint(elapsed_string, "cyan", end="\n\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 1

# %%
data_viz_w1_dumb = {}
path_w1 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w1)
    data_viz_w1_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    it_time = time_elapsed
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint("Dataset elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(it_time*(dumb_count-model_ind))

    cprint("[INFO]", "magenta", end=" ")
    cprint("Expected time left =", end=" ")
    cprint(elapsed_string, "cyan", end="\n\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 3

# %%
data_viz_w3_dumb = {}
path_w3 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, dumb_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='dumb', ds_path=path_w3)
    data_viz_w3_dumb[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    it_time = time_elapsed
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(time_elapsed)

    cprint("[INFO]", "magenta", end=" ")
    cprint("Dataset elapsed time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  
    
    now = datetime.now()
    time_elapsed = now - now_beg
    elapsed_string = humanize.precisedelta(it_time*(dumb_count-model_ind))

    cprint("[INFO]", "magenta", end=" ")
    cprint("Expected time left =", end=" ")
    cprint(elapsed_string, "cyan", end="\n\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint("Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %%
viz_all_dumb = {0: data_viz_clean_dumb, 1: data_viz_w15_dumb, 2: data_viz_w1_dumb, 3: data_viz_w3_dumb}

with open(f"Models/viz_all_dumb_{gen_date}.pickle", "wb") as f:
    pickle.dump(viz_all_dumb, f)
    

# %% [markdown]
# ### Smart models data generation

# %% [markdown]
# #### Clean data

# %%
smart_count = 30

# %%
data_viz_clean_smart = {}
path_clean = pathlib.Path("./Datasets/M=50/5000-1000-500/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_clean)
    data_viz_clean_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 0.15

# %%
data_viz_w15_smart = {}
path_w15 = pathlib.Path("./Datasets/M=50/W=0.15-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w15)
    data_viz_w15_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 1

# %%
data_viz_w1_smart = {}
path_w1 = pathlib.Path("./Datasets/M=50/W=1-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w1)
    data_viz_w1_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %% [markdown]
# #### W = 3

# %%
data_viz_w3_smart = {}
path_w3 = pathlib.Path("./Datasets/M=50/W=3-500-samples/")
now_beg = datetime.now()
now_last_it = datetime.now()
for model_ind in range(1, smart_count):
    viz_tmp = models_viz_datagen(samples_num=None, model_no=model_ind, model_type='smart', ds_path=path_w3)
    data_viz_w3_smart[model_ind] = viz_tmp
    now = datetime.now()
    time_elapsed = now - now_last_it
    elapsed_string = humanize.precisedelta(time_elapsed)
    now_last_it = datetime.now()

    cprint("[INFO]", "magenta", end=" ")
    cprint("Iteration time =", end=" ")
    cprint(elapsed_string, "cyan", end="\n")  

now = datetime.now()
time_elapsed = now - now_beg
elapsed_string = humanize.precisedelta(time_elapsed)

cprint("[INFO]", "magenta", end=" ")
cprint(" Total elapsed time =", end=" ")
cprint(elapsed_string, "cyan", end="\n")  

# %%
viz_all_smart = {0: data_viz_clean_smart, 1: data_viz_w15_smart, 2: data_viz_w1_smart, 3: data_viz_w3_smart}

# %%
with open(f"Models/viz_all_smart_{gen_date}.pickle", "wb") as f:
    pickle.dump(viz_all_smart, f)


