import os
import os.path as osp
import glob
import tqdm
import argparse
from typing import Dict

import mmcv
import mmengine
import numpy as np
from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.structures import DetDataSample
from mmdet.structures.mask import encode_mask_results, mask2bbox
from mmdet.apis import DetInferencer

try:
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None
    VOID = None

parser = argparse.ArgumentParser(description='Heatmap Inference')
parser.add_argument('config', help='Config file')
parser.add_argument('checkpoint', help='Checkpoint file')
parser.add_argument('--input', help='Image file/regex', type=str, required=True)
parser.add_argument('--output', help='Output directory', default='./output')
parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for inference', default=True)
args = parser.parse_args()

class HeatmapInferencer(DetInferencer):
    def __init__(self, config, checkpoint, device):
        super().__init__(config, checkpoint, device=device)

    def pred2dict(self,
                  data_sample: DetDataSample,
                  pred_out_dir: str = '') -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        is_save_pred = True
        if pred_out_dir == '':
            is_save_pred = False

        if is_save_pred and 'img_path' in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_img_path = osp.join(pred_out_dir, 'preds',
                                    img_path + '_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds', img_path + '.json')
        elif is_save_pred:
            out_img_path = osp.join(
                pred_out_dir, 'preds',
                f'{self.num_predicted_imgs}_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds',
                                     f'{self.num_predicted_imgs}.json')
            self.num_predicted_imgs += 1

        result = {}
        if 'pred_instances' in data_sample:
            masks = data_sample.pred_instances.get('masks')
            pred_instances = data_sample.pred_instances.numpy()
            result = {
                'labels': pred_instances.labels.tolist(),
                'scores': pred_instances.scores.tolist()
            }
            if 'bboxes' in pred_instances:
                result['bboxes'] = pred_instances.bboxes.tolist()
            if masks is not None:
                if 'bboxes' not in pred_instances or pred_instances.bboxes.sum(
                ) == 0:
                    # Fake bbox, such as the SOLO.
                    bboxes = mask2bbox(masks.cpu()).numpy().tolist()
                    result['bboxes'] = bboxes
                encode_masks = encode_mask_results(pred_instances.masks)
                for encode_mask in encode_masks:
                    if isinstance(encode_mask['counts'], bytes):
                        encode_mask['counts'] = encode_mask['counts'].decode()
                result['masks'] = encode_masks

        if 'pred_panoptic_seg' in data_sample:
            if VOID is None:
                raise RuntimeError(
                    'panopticapi is not installed, please install it by: '
                    'pip install git+https://github.com/cocodataset/'
                    'panopticapi.git.')

            pan = data_sample.pred_panoptic_seg.sem_seg.cpu().numpy()[0]
            pan[pan % INSTANCE_OFFSET == len(
                self.model.dataset_meta['classes'])] = VOID
            pan = id2rgb(pan).astype(np.uint8)

            if is_save_pred:
                mmcv.imwrite(pan[:, :, ::-1], out_img_path)
                result['panoptic_seg_path'] = out_img_path
            else:
                result['panoptic_seg'] = pan
        
        if "lam" in data_sample:                            
            lam = data_sample.lam.cpu().numpy()
            result['lam'] = lam.tolist()
        if "wh_map" in data_sample:
            wh_map = data_sample.wh_map.cpu().numpy()
            result['wh_map'] = wh_map.tolist()
        if "class_map" in data_sample:
            class_map = data_sample.class_map.cpu().numpy()
            result['class_map'] = class_map.tolist()
        

        if is_save_pred:
            mmengine.dump(result, out_json_path)
        

        return result

device = 'cuda:0' if args.use_cuda else 'cpu'
inferencer = HeatmapInferencer(args.config, args.checkpoint, device=device)

os.makedirs(args.output, exist_ok=True)
for p in tqdm.tqdm(glob.glob(args.input)):
    inferencer(p, out_dir=args.output, pred_score_thr=0.01, no_save_pred=False)
