NAME

intel_inference.py - Perform inference using a pre-trained model on a folder of 
images and generate predictions in a CSV file.


SYNOPSIS

python inference.py [-m MODEL_SCRIPT] [-i IMAGE_FOLDER] [-o OUTPUT]


DESCRIPTION

The `inference.py` script allows you to perform inference using a pre-trained 
model on a folder of images and generate predictions in a CSV file. The script 
loads the model, processes the images, and writes the predictions to the 
specified output file.


OPTIONS

* `-m MODEL_SCRIPT`, `--model-script MODEL_SCRIPT`: Specifies the path to the 
  model script. This is the pre-trained model that will be used for inference. 
  The default path is `vit_20_model_script_cuda.pt`.

* `-i IMAGE_FOLDER`, `--image-folder IMAGE_FOLDER`: Specifies the path to the 
  folder containing the images for inference. The script will process all the 
  `.jpg` files in the specified folder. 
  The default path is `../../ADEIJ_datasets/seg_pred/seg_pred`.

* `-o OUTPUT`, `--output OUTPUT`: Specifies the name of the output CSV file. 
  The predictions will be written to this file. If the file already exists, 
  it will be overwritten. 
  The default file name is `preds.csv`.

* `-d`, `--device`: Print the currently available torch device instead of 
  running the inference script.


EXAMPLES

1. Perform inference using the CPU model script and default output file, with a 
   custom image folder:

   python intel_inference.py -m vit_20_model_script_cpu.pt -i test

2. Perform inference using the CUDA model script, custom image folder, and 
   custom output file:

   python intel_inference.py -m vit_20_model_script_cuda.pt -i my_test/ -o my_preds.csv


NOTES

- See ../requirements.txt for Python packages required to run this script

- It is critical you select the correct option for your device type, else the 
  TorchScript will not load correctly. If you are unsure what device type you 
  have, run  `python intel_inference.py -d` to print your device to the console.

  - If it isn't obvious, the correct model to use is:
      device == 'cuda' -> use model vit_20_model_script_cuda.pt
      device == 'cpu'  -> use model vit_20_model_script_cpu.pt

- The script assumes that the images in the image folder are in JPEG format 
  (`.jpg` extension). Only files with the `.jpg` extension will be processed.

- The script expects the model script to be in the TorchScript format (`.pt` 
  file). See: https://pytorch.org/docs/stable/jit.html


/!\ DEVICE TYPE /!\

In the submission, two models were included. One is meant for Nvidia GPU's and 
one is for CPU. 

If your system has a cuda compatible nvidia GPU, please ignore the following as 
if you have successfully installed the correct packages and their correct 
versions, there should be no need to use the CPU model.

GPU models are models that were saved using torch.jit while the device was in 
the GPU hence if a system does not have a cuda compatible NVIDIA GPU the model 
cannot be loaded properly. Hence another model that was moved to the CPU before 
saving was created as well to make sure a model can be used. For more 
information, please visit https://github.com/pytorch/pytorch/issues/33212

If you do not have a cuda compatible GPU and the script does not run, please 
download the following packages after uninstalling your current distribution:

torch==2.0.1+cpu
torchaudio==2.0.2+cpu
torchvision==0.15.2+cpu

and run the inference script as is.


/!\ DEBUG NOTE /!\

If the following errors appear when running the inference script:
    aten::empty_strided
    aten::scaled_dot_product_attention

please uninstall your current torch distribution and

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(for pytorch 2.0.1+cu118)

and try running the inference again. 

This error happens when incompatible versions of torch are installed.


--------------------------------------------------------------------------------

On our pseudo-test set of 3000 labelled images from the unlabelled seg_pred 
directory, the GPU model achieved an accuracy of 85.77% and the CPU model 
achieved an accuracy of 84.53%