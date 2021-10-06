# CTNet
Official Pytorch implementation of the paper "[Contextual Transformation Network for Lightweight Remote Sensing Image Super-Resolution](https://github.com/BITszwang/CTNet/)".

## Requirements

- Python 3.7
- Pytorch=1.5
- torchvision=0.6.0 
- matplotlib
- opencv-python
- scipy
- tqdm
- scikit-image

## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```

## Dataset
We used the UCMerced dataset for both training and test. Please first download the dataset via [Baidu Drive](https://pan.baidu.com/s/1XiFhJT9eExfebV3TSkjY2w) (key:912V). 

## Train
The train/val data pathes are set in [data/__init__.py](codes/data/__init__.py) 
```
# x4
python demo_train_ctnet.py --model=CTNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=CTNETx4_UCMerced
# x3
python demo_train_ctnet.py --model=CTNET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=CTNETx4_UCMerced
# x2
python demo_train_ctnet.py --model=CTNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=CTNETx4_UCMerced
```



## Test 
The test data path and the save path can be edited in [demo_deploy.py](codes/demo_deploy_ctnet.py)

```
# x4
python demo_deploy_ctnet.py --model=CTNET --scale=4
# x3
python demo_deploy_ctnet.py --model=CTNET --scale=3
# x2
python demo_deploy_ctnet.py --model=CTNET --scale=2
```

## Evaluation 
Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```

Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```

## Acknowledgements 
This code is built on [HSENet (Pytorch)](https://github.com/Shaosifan/HSENet). We thank the authors for sharing the codes.  
