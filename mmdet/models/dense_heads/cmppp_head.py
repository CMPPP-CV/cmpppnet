import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from .base_dense_head import BaseDenseHead
from ..utils import get_topk_from_heatmap, transpose_and_gather_feat


@MODELS.register_module()
class CMPPPHead(BaseDenseHead):
    """The head used in CMPPPNet for object classification and box regression.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        loss_center_heatmap (ConfigType): Config for center heatmap loss.
        loss_wh (ConfigType): Config for width-height regression loss.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 loss_center_heatmap: ConfigType = dict(type='PPPLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_classification: ConfigType = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_wh: ConfigType = dict(type='L1Loss', loss_weight=0.1),
                 init_cfg=None,
                 pooling_size=4,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_classification = MODELS.build(loss_classification)
        self.loss_wh = MODELS.build(loss_wh)
        self.pooling_size = pooling_size
        self.fp16_enabled = False

    def forward(self, x):
        """Forward function."""
        return (x[:, 0, ...], # center heatmap predictions
                x[:, 1:3, ...],  # width-height predictions
                x[:, -self.num_classes:, ...]               # class predictions
                )
    
    def get_targets(self, gt_bboxes, gt_labels,
                    feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (tuple): feature map shape with value [B, _, H, W]
            img_shape (tuple): image shape.

        Returns:
            tuple[dict, float]: The float value is mean avg_factor, the dict
            has components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                # radius = gaussian_radius([scale_box_h, scale_box_w],
                #                          min_overlap=0.3)
                # radius = max(0, int(radius))
                ind = gt_label[j]
                center_heatmap_target[batch_id, ind, cty_int, ctx_int] = 1
                # gen_gaussian_target(center_heatmap_target[batch_id, ind],
                #                     [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor
    
    def predict(self, x, batch_data_samples, rescale = False):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x) # (lambda, wh map, class map)
        lam = torch.exp(outs[0]) / torch.tensor(self.loss_center_heatmap.test_resolution).prod()

        # Determine expected number of objects and select local maxima
        num_predictions = torch.sum(lam)
        kernel = torch.ones((1, 1, self.pooling_size, self.pooling_size), device=x.device)
        pooled_intensity = F.conv2d(lam.unsqueeze(1), kernel, stride=self.pooling_size)
        scores, indices, _, cy, cx = get_topk_from_heatmap(pooled_intensity, k=int(num_predictions.item()))

        # Gather and determine extent of predicted bounding boxes
        wh_map = outs[1]
        wh = transpose_and_gather_feat(
            F.avg_pool2d(wh_map, self.pooling_size, self.pooling_size), indices
        )

        cx = cx * self.pooling_size + self.pooling_size // 2
        cy = cy * self.pooling_size + self.pooling_size // 2
        tl_x = cx - wh[..., 0] / 2
        tl_y = cy - wh[..., 1] / 2
        br_x = cx + wh[..., 0] / 2
        br_y = cy + wh[..., 1] / 2

        # Assemble bounding boxes and rescale to original image size if necessary
        batch_bboxes = torch.cat([tl_x, tl_y, br_x, br_y],dim=0).permute(1, 0)
        img_meta = batch_img_metas[0]
        if rescale and 'scale_factor' in img_meta:
            batch_bboxes[..., :4] /= batch_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        
        # Determine PPP analogue of objectness score
        objectness = scores.clamp(0, 1)

        # Determine predicted classes
        class_map = F.softmax(outs[2], dim=1)
        classes = transpose_and_gather_feat(
            F.avg_pool2d(class_map, self.pooling_size, self.pooling_size), indices
        )
        labels = classes.argmax(dim=2)

        # Assemble predicted instance objects
        results = InstanceData()
        results.bboxes = batch_bboxes
        results.scores = objectness.squeeze()
        results.labels = labels.long().squeeze()

        return results, lam, wh_map, class_map
    
    def loss_by_feat(
        self, 
        center_heatmap_preds,
        wh_preds,
        class_preds,
        batch_gt_instances, 
        batch_img_metas, 
        batch_gt_instances_ignore = None
        ):
        gt_bboxes = [
            gt_instances.bboxes for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0]['batch_input_shape']
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     wh_preds.shape,
                                                     img_shape)
        center_heatmap_target = target_result['center_heatmap_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        wh_target = target_result['wh_target']
        
        loss_center = self.loss_center_heatmap(
            center_heatmap_preds,
            center_heatmap_target,
            # weight=wh_offset_target_weight[:, 0, ...][:, None, ...],
            avg_factor=avg_factor
        )

        loss_cls = self.loss_classification(
            class_preds,
            center_heatmap_target,
            weight=wh_offset_target_weight[:, 0, ...][:, None, ...],
            avg_factor=avg_factor
        )
        
        loss_wh = self.loss_wh(
            wh_preds,
            wh_target,
            weight=wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        
        return dict(loss_center=loss_center, loss_wh=loss_wh, loss_cls=loss_cls)