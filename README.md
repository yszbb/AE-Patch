# AE-Patch
Official Pytorch implementation for paper "Adversarial Event Patch for Spiking Neural Networks".
![Figure](https://github.com/yszbb/AE-Patch/blob/main/asserts/pipeline.jpg)
## Requirements
- python 3.8
- Pytorch 1.13
- At least 1x12GB NVIDIA GPU
## Installation
```
git clone https://github.com/yszbb/AE-Patch
cd AE-Patch
pip install -r requirements.txt
```
## Preparation
### Dataset
1. Download .
2. Filter out .
3. We have .
4. Put the .
### Model
1. Download the .
2. Fine-tune .
### train and val
Once you have setup your path, you can run an experiment like so:
```
python patch_untargeted_everyone.py --arch "vggdvs" --datadvs "dvscifar" 
```
The terminal will print the gbest_position and gbest_value.
