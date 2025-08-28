# This code is essnetially taken from the CenterNet implementation
from torch import Tensor
from typing import Union

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class CMPPPNet(SingleStageDetector):
    """Implementation of CMPPPNet detector."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        x = self.extract_feat(batch_inputs)
        results, lam, wh_map, class_map = self.bbox_head.predict(
            x, 
            batch_data_samples, 
            rescale=rescale
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, [results])
        batch_data_samples[0].set_data({'lam': lam, 'wh_map': wh_map, 'class_map': class_map})
        
        return batch_data_samples
    
    # def loss(self, batch_inputs: Tensor,
    #          batch_data_samples: SampleList) -> Union[dict, list]:
    #     """Calculate losses from a batch of inputs and data samples.

    #     Args:
    #         batch_inputs (Tensor): Input images of shape (N, C, H, W).
    #             These should usually be mean centered and std scaled.
    #         batch_data_samples (list[:obj:`DetDataSample`]): The batch
    #             data samples. It usually includes information such
    #             as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     x = self.backbone._forward(batch_inputs, batch_data_samples)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     losses = self.bbox_head.loss(x, batch_data_samples)
    #     return losses
    