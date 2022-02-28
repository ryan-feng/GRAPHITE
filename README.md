# GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems
Code for [GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems](https://arxiv.org/abs/2002.07088), to appear in the 2022 IEEE European Symposium on Security and Privacy (EuroS&P).

## Prereqs 
### Packages
```
matplotlib==3.1.2
numpy==1.17.4
prettytable==2.2.0
opencv_python==4.2.0.32
pandas==0.25.3
torchvision==0.8.1
joblib==0.14.0
tqdm==4.60.0
scipy==1.3.3
torch==1.7.0
kornia==0.5.10
Pillow==9.0.0
python_Levenshtein==0.12.2
```
### Setup
Run `./make_output_folders.sh` to setup the output directory structure. GTSRB attacks will be saved in `outputs/boosted` and GTSRB masks will be saved in `outputs/masks`. CIFAR-10 attacks will be saved in `outputs/CIFAR/boosted`. OpenALPR attacks will be saved in `outputs/OpenALPRBorder/boosted`.

## Running GTSRB attacks:

Attacks typically take ~10 minutes on an RTX 3080 GPU.
### General usage:
```
python3 main.py -v <victim class index> -t <target class index> --tr_lo <tr_lo> --tr_hi <tr_hi> -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```
### Running GTSRB attacks from Table 8:
Stop sign to Speed Limit 30: <br>
```
python3 main.py -v 14 -t 1 --tr_lo 0.65 --tr_hi 0.85 -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```
Stop sign to Pedestrians: <br>
```
python3 main.py -v 14 -t 27 --tr_lo 0.65 --tr_hi 0.85 -s score.py -n GTSRB --heatmap=Target --coarse_mode=binary -b 100 -m 100
```

Example outputs from Table 8 included in `example_outputs`. Results may vary depending on GPU / Nvidia / CUDA setup.

## Running CIFAR-10 attacks:

### Prereq: set up data and model
```
cd cifar;
python3 create_cifar_vts.py;
```
Download [weights](https://drive.google.com/file/d/1douNo27f68EMftJYDG84vE6ThkgDGhj2/view?usp=sharing) and place in the `cifar/` folder.

### General usage:
```
python3 main.py -v <class index of victim> -t <class index of target> --img_v <victim image path> --img_t <target image path> --hull cifar/masks/mask.png -s score.py -n CIFAR --heatmap=Target --coarse_mode=binary -b 100 -m 100 --image_id <image id for output purposes> 
```

## Running ALPR attacks:
Attacks typically take ~2 hrs on an RTX 3080 GPU, but the vast majority of the time is just writing out and reading back in the image in JPG format to pass through the ALPR tool on queries, not the actual attack computation. Attack by default writes out and loads to `temp.jpg`. If you want to change this, simply change line 17 of `OpenALPR/OpenALPRBorderNet.py`.

### Prereq: Install OpenALPR
Follow the appropriate instructions for your OS from this link: https://github.com/openalpr/openalpr/wiki. On our machine, the instructions in the Ubuntu Linux page in the section titled "**The Easy Way**" worked well. <br>


### General usage:
```
python3 main_alpr.py -v 0 -t 1 --tr_lo <tr_lo> --tr_hi <tr_hi> -s score_border.py -n OpenALPRBorder --heatmap=Target --coarse_mode=binary -b 50 -m 10 --vic_license_plate=<victim plate> --img_v=<victim image file> --img_t=<target image file> --border_outer=<mask of license plate holder> --border_inner=<mask of inside of the license plate holder> --bt --tag=<tag to save with outputs> --pt_file=
```
### Running ALPR attacks from Table 12:
Mazda: <br>
```
python3 main_alpr.py -v 0 -t 1 --tr_lo .2 --tr_hi .6 -s score_border.py -n OpenALPRBorder --heatmap=Target --coarse_mode=binary -b 50 -m 10 --vic_license_plate=BAU7299 --img_v=inputs/OpenALPR/mazda.png --img_t=inputs/OpenALPR/mazda_noplate.png --border_outer=inputs/OpenALPR/mazda_border_outer.png --border_inner=inputs/OpenALPR/mazda_border_inner.png --bt --tag=mazda --pt_file=
```
Camry: <br>
```
python3 main_alpr.py -v 0 -t 1 --tr_lo .2 --tr_hi .6 -s score_border.py -n OpenALPRBorder --heatmap=Target --coarse_mode=binary -b 50 -m 10 --vic_license_plate=BGP9112 --img_v=inputs/OpenALPR/camry.png --img_t=inputs/OpenALPR/camry_noplate.png --border_outer=inputs/OpenALPR/camry_border_outer.png --border_inner=inputs/OpenALPR/camry_border_inner.png --bt --tag=camry --pt_file=
```
Example outputs from Table 12 included in `example_outputs`. Results may vary depending on GPU / Nvidia / CUDA setup.

## Tuning GRAPHITE:
Key parameters than can be tuned based on the ablations in Section 6.3:
```
--tr_lo <x>: sets the minimum transform-robustness threshold to x for fine-grained_reduction
--max_mask_size <x>: adds the max mask option and sets it to x. If used, you should also set tr_lo = 0
-m <x>: uses x transforms in mask generation
-b <x>: uses x transforms in boosting
--coarse_mode [none | binary | linear]: sets type of coarse grained reduction. Set to 'none' to disable. Default is 'binary'
--heatmap [Target | Victim | Random]: changes the type of heatmap
--joint_iters <x>: run x alternations between mask generation and boosting
```

## Other variants from the paper:
### White-box Attacks:
```
cd whitebox;
python3 whitebox_attack.py --victim <victim_index> --target <target_index> --out <out path>
python3 whitebox_patch.py --victim <victim_index> --target <target_index> --mask_size 64 --out <out path>
```

### Baselines:
```
cd baselines;
python3 l0_and_opt_normal.py --victim_label=<victim label> --target_label=<target label> --victim_img_path=<victim img path> --target_img_path=<target img path> --initial_mask_path=<mask img path> --xforms_pt_file=<pt file>
python3 l0_and_opt_eot.py --victim_label=<victim label> --target_label=<target label> --victim_img_path=<victim img path> --target_img_path=<target img path> --initial_mask_path=<mask img path> --xforms_pt_file=<pt file>
python3 l0_and_boosting_no_tolerance.py --victim_label=<victim label> --target_label=<target label> --victim_img_path=<victim img path> --target_img_path=<target img path> --initial_mask_path=<mask img path> --xforms_pt_file=<pt file>
```

### Patchguard:
Setup: 
```
cd patchguard;
./make_output_folders.sh;
python3 create_cifar_vts.py;
git clone https://github.com/inspire-group/PatchGuard.git;
cd PatchGuard;
git checkout ae23629644d8628ef376d3121c35de261472daf5;
```
Finally, download the `bagnet17_192_cifar.pth` checkpoint from the PatchGuard repo [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT) and place it in the `PatchGuard/checkpoints/` folder. <br>

Running:
```
cd patchguard;
python3 main_patchguard.py
```

### Citation
```
@article{feng2020graphite,
  title={GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems},
  author={Feng, Ryan and Mangaokar, Neal and Chen, Jiefeng and Fernandes, Earlence and Jha, Somesh and Prakash, Atul},
  journal={arXiv preprint arXiv:2002.07088},
  year={2020}
}
```

### References
[1] OPT-attack: https://github.com/LeMinhThong/blackbox-attack <br>
[2] PatchGuard: https://github.com/inspire-group/PatchGuard <br>
