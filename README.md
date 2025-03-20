# Denoising
This repository allows to perform training and inference for **multiplexed image denoising** using a U-Net based CNN.
The network is meant to be trained in a supervised regime, using a ground truth clean images. 
## Installation
```
conda env create -f environment.yml
conda activate denoise
```
## Usage 
To use this repository, you will need to edit the config files at config\config_inference.yaml or config\config_train.yaml
### Inference 
Currently inference is performed only on a single protein.
The arguments in the config file:
**results_folder_name**: name of the folder to save the results to (will appear under a "results" dir)

**test_dataset_path**: Path to a directory containing the input images. The expected structre is very specific: test_dataset_path\noisy\sub_dataset\TIFs\images.tif
note that currently the test directory must also include clean images under test_dataset_path\clean\sub_dataset\TIFs\images.tif
(This doesn't realy makes sense, sorry)

**sub_dataset**: A list of directory names to be used to access the test data (see previous argument)

**protein**: string, the name of the protein to perform inference on.

**image_normalization**: Normalization type, use deafult value (min_max)

**model_type**: Model type, use deafult value (AttUNet)

**model_name**: name of the model file to be used (just the name without .pt sufix), should be saved in this dir under "pretrained_models" dir

**confidence_output** whether to output the float prediction value for each pixel, or a mask of True False

**center_crop** Leave empty to perform inference on the full images


### Training
Very similiar arguments to the inference, plus a bunch of training settings that are pretty self explanatory.

## Trained models
An ensemble of trained models can be found on wexac under 
Labs\Leeat\Collaboration\Denoising\supervised
| Model purpose    | sub path |
| -------- | ------- |
| MIBI General denosier  |denoising147-general_denoiser\pretrained_models\Denoising147-90000.pt   |
| Fine tune for specific proteins (MIBI) | denoising148-fine_tuning_protein\pretrained_models\protein-name.pt     |
| CODEX General model    | denoising149_CODEX\pretrained_models\Denoising149-20000.pt   |
|Fine tune for specific proteins (CODEX)|denoising150_ft_CODEX\pretrained_models\protein-name-latest.pt|


Note that fine tuned models exist only for some subset of the proteins. 

## Notes and acknowledgements
- This project was created by the awesome **Lior Ben Shabat**
- This readme file was created by Eli Ovits, 20.3.25
- If you run into any issues, the most helpful person to ask is probably Tamar Kashti from the AI hub