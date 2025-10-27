Code for the paper "Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements".

The code was executed using python 3.12.8, torch 2.6.0 and deepinv 0.3.3.

## Datasets

The datasets (except for the MRI scans which are extracted from FMRI) are available at ... and should be put in the datasets directory.

## Models

The models used in each experiment are available at ... and should be put in:
- models (for kernel selection experiments)
- models/diffusion (for OOD detection experiments).

The GSDRUnet grayscale weights can be downloaded using deepinv.

The FFHQ and AFHQ trained models come from: https://github.com/jychoi118/ilvr_adm

The MRI models, as well as the code contained in directories EDM and torch_utils
come from https://github.com/wustl-cig/Measurement-domain-KL-divergence


## Kernel selection experiments

Use kernel_comparison.py to launch kernel selection experiments:
```
python3 kernel_comparison.py i 0 2
```
computes the samples for the 2 test images, for the ground truth kernel number i.

The notebook kernel_comparison.ipynb can then be used to evaluate the kernel selection accuracy using the saved samples.

kernel_comparison_reconstruction.ipynb can be used to compute the likelihood on MAP to compare to our method.
kernel_comparison_SAPG.py is used to fit the regularization parameter for each kernel/test image.

## OOD detection experiments

Use compute_samples_natural.py to generate samples for prior misspecification testing on natural images, and compute_samples_mri.py for MRI experiments:
```
python3 compute_samples_natural.py "results/diffusion/natural/FFHQ_FFHQ_05" 0 74
```
to compute the samples indexed 0 to 74 for the experiment described in "results/diffusion/natural/FFHQ_FFHQ_05/config.json". The samples are saved in the same directory as the config file.
The samples can then be post-processed using OOD_detection.ipynb to compute the metrics.

## Gaussian Analytical case

The analytical test cases are implemented in test_evaluation.py.