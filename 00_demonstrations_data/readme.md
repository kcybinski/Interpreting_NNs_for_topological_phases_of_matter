The large files (models, datasets, 100_CNNs) can be downloaded from code release on Zenodo, and after unzipped, they should form such folder structure:   

 📦 00_demonstrations_data   
 ┣ 📂 100_CNNs   
 ┃ ┗ 📂 bs=64_perturb=True_lr=1e-04   
 ┣ 📂 datasets   
 ┃ ┣ 📂 disordered_W=0.01   
 ┃ ┣ 📂 disordered_W=0.05   
 ┃ ┣ 📂 disordered_W=0.15   
 ┃ ┣ 📂 disordered_W=0.5   
 ┃ ┣ 📂 disordered_W=1.0   
 ┃ ┣ 📂 disordered_W=2.0   
 ┃ ┣ 📂 disorderless   
 ┃ ┗ 📜 targets_25_reals.pkl   
 ┣ 📂 figures   
 ┣ 📂 images   
 ┃ ┗ 📜 architecture.png   
 ┣ 📂 models   
 ┃ ┣ 📜 cams_all_toy_model.pkl   
 ┃ ┣ 📜 parameters.txt   
 ┃ ┣ 📂 poorly_generalizing_good_CAM   
 ┃ ┣ 📂 poorly_generalizing_misleading_CAM   
 ┃ ┣ 📜 readme.md   
 ┃ ┣ 📜 targets_25_reals.pkl   
 ┃ ┣ 📂 well_generalizing_good_CAM   
 ┃ ┗ 📂 well_generalizing_misleading_CAM   
 ┣ 📜 all_stats_10_real_100_slices.csv   
 ┗ 📜 readme.md   