import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TrayFoodDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'tray', 'cutlery', 'form', 'straw', 'meatball', 'beef', 'roast lamb', 'beef tomato casserole', 'ham', 'bean', 'cucumber', 'leaf', 'tomato', 'boiled rice', 'beef mexican meatballs', 'spinach and pumpkin risotto', 'baked fish', 'gravy', 'zucchini', 'carrot', 'broccoli', 'pumpkin', 'celery', 'sandwich', 'side salad', 'tartare sauce', 'jacket potato', 'creamed potato', 'bread', 'margarine', 'soup', 'apple', 'canned fruit', 'milk', 'vanilla yogurt', 'jelly', 'custard', 'lemon sponge', 'juice', 'apple juice', 'orange juice', 'water')

    PALETTE = [[192, 192, 192], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128]]

    def __init__(self, split, **kwargs):
        super(TrayFoodDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir)