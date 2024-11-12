<p align="center">
    <img src="https://www.jaejoonglee.com/images/rgb2point.png" alt="Overview">
</p>

**RGB2Point** is officially accepted to WACV 2025. It takes a single unposed RGB image to generate 3D Point Cloud. Check more details from the [paper](https://arxiv.org/pdf/2407.14979).
## Codes
**RGB2Point** is tested on Ubuntu 22 and Windows 11. `Python 3.9+` and `Pytorch 2.0+` is required.

## Dependencies
Assuming `Pytorch 2.0+` with `CUDA` is installed, run:
```
pip install timm
pip install accelerate
pip install wandb
pip install open3d
pip install scikit-learn
```

## Training
```
python train.py
```

## Pretrained Model
Download the model trained on Chair, Airplane and Car from ShapeNet.
```
https://drive.google.com/file/d/1Z5luy_833YV6NGiKjGhfsfEUyaQkgua1/view?usp=sharing
```

## Inference
```
python inference.py
```
Change `image_path` and `save_path` in `inference.py` accrodingly.




## Reference
If you find this paper and code useful in your research, please consider citing:
```bibtex
@article{lee2024rgb2point,
  title={RGB2Point: 3D Point Cloud Generation from Single RGB Images},
  author={Lee, Jae Joong and Benes, Bedrich},
  journal={arXiv preprint arXiv:2407.14979},
  year={2024}
}
```