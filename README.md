# SALENet
This is the project page for paper "SALENet: Structure-Aware Lighting Estimations from a Single Image for Indoor Environments"
To run the code, follow the following 3 steps:

**Step 1: Generate Spherical Guassian (SG) lighting priors

cd SG_Test

python sg_test_main.py

**Step 2: Generate Spherical Harmonics(SH) lighting priors

cd SH_Test

pyton sh_test_main.py

**Step 3: Environment Map generation based on generated SG and SH lighting priors

cd Env_Gen

python env_gen_main.py

The model can be downloaded here [https://drive.google.com/drive/folders/1OyzkgH3Kn9fgdtO8b7YyO4NEOknbP_Us?usp=drive_link](https://vuw-my.sharepoint.com/:f:/g/personal/zhaoj1_staff_vuw_ac_nz/EopvbIaay0xPgsv66Erq6j4BKdq0feVVI0PldJnwhCfITQ?e=QwbBaa)

After downloading, put the models to each model folder according to their names.
