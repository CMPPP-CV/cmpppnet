from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class TirodDataset(CocoDataset):
    METAINFO = {
        'classes': (
            'bag', 'bottle', 'cardboard box', 'chair', 'potted plant', 'traffic cone', 'trashcan', 'ball', 'broom', 'garden hose', 'bucket', 'bycicle', 'gardening tool'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175)]
         }