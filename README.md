# Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection

[![Paper](https://img.shields.io/badge/paper-arXiv%20%7C%20Journal-blue)](https://arxiv.org/abs/2506.21486)

This repository provides the **official implementation** for the paper:

> **Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection**  
> Tobias J. Riedlinger, Kira Maag, Hanno Gottschalk
> *ICLR, 2026*

The code is based on MMDetection v3.3.0 intended to support **reproducibility**, **benchmarking**, and **further research and development** based on the published work.

---

## 📄 Paper

- **Title:** Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection
- **Authors:** Tobias J. Riedlinger, Kira Maag, Hanno Gottschalk
- **Venue:** International Conference on Learning Representations (ICLR), 2026  
- **Link:** [https://arxiv.org/abs/2506.21486](https://arxiv.org/abs/2506.21486)

If you use this code in your research, please cite the paper (see [Citation](#citation)).

---

## 📦 Repository Structure
The repository generally integrates the CMPPP model into the MMDetection framework. The central addition is the folder `/evaluation` which contains the implementation of the evaluation scripts.

## Usage
To use the code, please follow the instructions in the [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/). The evaluation scripts for the CMPPP model can be found in the `/evaluation` folder.

For evaluation, edit all necessary paths stored in `evaluation/global_defs.py` and select the tasks to be executed by setting the corresponding boolean variable (`True/False`). Run the evaluation code:
```bash
python evaluation/evaluate.py
```


# Citation
```
@inproceedings{
riedlinger2026towards,
title={Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection},
author={Tobias Riedlinger and Kira Maag and Hanno Gottschalk},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=M2KLWLHzX0}
}
```