# Lab 1: Reproducing DeepLabV3+'s record
## - Reproduced results
### Cityscapes
| Source     | #GPUs | Method | Backbone | Crop Size | Lr schd |mIoU|Config|training_notebook|
| ------------ | :---: | :----: | -------: | --------- | -------|----:|--------|:---------------|
| Official   | 4    | DeepLabV3+ | R-50-d8 | 512x1024 | 40000  |79.61|[Config](configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py)|/|
| Reproduced  | 2    | DeepLabV3+ | R-50-d8 | 512x1024 | 80000  |79.29|[Config](configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py)|[Notebook](ipynb/deeplab_cityscapes.ipynb)|

--------------------

# Lab2: Training on CamVid dataset
###  CamVid
| Source     | #GPUs | Method | Backbone | Crop Size | Lr schd |mIoU|Config|training_notebook|
| ------------ | :---: | :----: | -------: | --------- | ------- |----:|--------|:---------------|
| Reproduced   | 2    | DeepLabV3+ | R-50-d8 | 640x640 | 80000  |76.18|[Config](configs/deeplabv3plus/deeplabv3plus_r50-d8_640x640_80k_camvid.py)|[Notebook](ipynb/camvid_deeplab+.ipynb)|
| Reproduced  | 2    | FCN | Unet | 640x640 | 80000  |72.33|[Config](/configs/unet/fcn_unet_r5-d16_640x640_80k_camvid.py)|[Notebook](ipynb/camvid_unet.ipynb)|
#
