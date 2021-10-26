None
None
# Lab 1: Reproducing DeepLabV3+'s record
## - Reproduced results
### Cityscapes
| Source     | #GPUs | Method | Backbone | Crop Size | Lr schd |  mIoU |
| ------------ | :---: | :----: | -------: | --------- | ------- | ----: |
| Official   | 4    | DeepLabV3+ | R-50-d8 | 512x1024 | 40000  | 79.61 |
| Reproduced  | 2    | DeepLabV3+ | R-50-d8 | 512x1024 | 80000  | 79.29 |

--------------------

# Lab2: Training on CamVid dataset
###  CamVid
| Source     | #GPUs | Method | Backbone | Crop Size | Lr schd |  mIoU |
| ------------ | :---: | :----: | -------: | --------- | ------- | ----: |
| Reproduced   | 2    | DeepLabV3+ | R-50-d8 | 640x640 | 80000  | 76.18 |
| Reproduced  | 2    | FCN | Unet | 640x640 | 80000  | 72.33 |

```{.python .input}

```
