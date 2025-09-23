import os
import os.path as osp
import glob
import tqdm
import time
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
parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for inference', default=False)
args = parser.parse_args()

device = 'cuda:0' if args.use_cuda else 'cpu'
inferencer = DetInferencer(args.config, args.checkpoint, device=device)

os.makedirs(args.output, exist_ok=True)
time_list = []
for p in tqdm.tqdm(glob.glob(args.input)):
    start_time = time.time()
    inferencer(p, draw_pred=False)
    end_time = time.time()
    time_list.append(end_time - start_time)
print(f'Average inference time: {np.mean(time_list):.4f} seconds')
print(f'FPS: {1/np.mean(time_list):.2f} frames per second')