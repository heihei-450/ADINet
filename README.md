# ADINet: Adaptive Dynamic Interaction for RGB-D Salient Instance Segmentation

<p align="center">
  <strong>Adaptive Dynamic Interaction for RGB-D Salient Instance Segmentation</strong>
</p>

<p align="center">
  <a href="https://github.com/heihei-450/ADINet">Code</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#experimental-results">Results</a>
</p>

## Overview

**ADINet** is an RGB-D salient instance segmentation framework designed to identify and segment individual salient objects by jointly exploiting RGB appearance information and depth geometry. It addresses three challenges: insufficient cross-modal interaction, inaccurate fine-grained instance boundaries, and unreliable depth information.

The framework adopts a dual-stream RGB/depth encoder with a deformable Transformer and two complementary prediction branches: a **Kernel Branch** for instance-aware kernels and a **Mask Branch** for pixel-level mask features. The resulting instance kernels and mask features are combined for salient instance segmentation.

## Architecture

The overall pipeline consists of:

1. **Dual-stream encoder:** separate RGB and depth feature extraction followed by deformable Transformer encoding.
2. **Mask Branch:** CMAE exchanges attention between modalities and constructs fused pixel-level mask representations.
3. **Kernel Branch:** FADE enhances depth representations and ADIK generates dynamic instance-aware kernels.
4. **Instance prediction:** instance kernels interact with mask features; training uses bipartite matching.

## Repository Structure

```text
ADINet/
├── ADINet/                 # Model, dataset and supporting implementation
├── ADINet.yaml             # Experiment configuration
├── train_net.py            # Training / evaluation entry point
├── infer_rgbd_sis.py       # RGB-D inference script
├── get_flops.py            # Complexity analysis
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/heihei-450/ADINet.git
cd ADINet
```

Create a Python environment and install the repository dependencies:

```bash
conda create -n adinet python=3.9 -y
conda activate adinet

# Install a compatible PyTorch / torchvision build for your CUDA setup first.
# See: https://pytorch.org/get-started/locally/

# Install a Detectron2 version compatible with your PyTorch / CUDA setup.
# See: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

pip install -r requirements.txt
```

**Note:** Python 3.9 is a suggested starting point, not a verified requirement. The repository's `requirements.txt` does not pin PyTorch, torchvision, CUDA, or Detectron2 versions. Ensure that those components are mutually compatible.

## Datasets

The paper uses the following RGB-D salient instance segmentation benchmarks:

| Dataset | Usage | Details |
|---|---|---|
| COME15K training split | Training | 8,025 RGB-D images |
| COME15K-E | Evaluation | 4,600 easier samples |
| COME15K-H | Evaluation | 3,000 harder samples |
| DSIS | Evaluation | 1,940 RGB-D image pairs in the dataset |
| SIP | Evaluation | 929 high-resolution RGB-D images |

Prepare RGB images, aligned depth maps, and instance annotations in the format expected by the dataset registration and mapper code in `ADINet/`. Update dataset paths in that registration code for your local machine.

The configuration `ADINet.yaml` references these registered dataset names:

```yaml
DATASETS:
  TRAIN: ("come15k_train",)
  TEST: ("come15k_test_e", "come15k_test_h", "dsis_test", "sip_test")
```

### Dataset Download Links

The datasets can be downloaded from the following links provided in the manuscript's **Data Availability** statement:

| Dataset | Download |
|---|---|
| COME15K | [Google Drive](https://drive.google.com/file/d/15UdZBdtmwxq7PiCHUpkMgnUxbLF1p_O5/view) |
| DSIS | [Google Drive](https://drive.google.com/file/d/1zqWo6vSB4ARJqu49PqoyKatj9xxolsPp/view) |
| SIP | [Google Drive](https://drive.google.com/file/d/1ebNjyrS28vEXDGawxHxVFNNxz3XLBqrT/view) |

**Note:** The links above were supplied in the Data Availability statement and have not been independently checked for accessibility. After downloading, check the dataset registration implementation before preparing files; do not assume a generic COCO folder layout is accepted without conversion.

## Training

The repository includes `train_net.py` and `ADINet.yaml`. A standard Detectron2-style invocation is:

```bash
python train_net.py --config-file ADINet.yaml
```

The paper reports the following training setup:

| Setting | Value |
|---|---|
| Framework | Detectron2 |
| Backbone | ImageNet-pretrained ResNet-50 |
| RGB/depth Transformer layers | 3 / 3 |
| Batch size | 4 |
| Initial learning rate | 2.5e-5 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Training iterations | 100,000 |
| Input resize | Short side 320, long side at most 480 |
| Reported hardware | Single NVIDIA RTX 3060 |

The repository configuration additionally specifies checkpoints every 20,000 iterations and 50 instance kernels.

**Before training:** confirm dataset registration, pretrained weights, and the actual CLI options supported by your checkout. The command above follows the typical Detectron2 interface and has not been execution-tested in this environment.

## Evaluation

After training, a standard Detectron2-style evaluation invocation is:

```bash
python train_net.py --config-file ADINet.yaml --eval-only MODEL.WEIGHTS /path/to/model_final.pth
```

Replace `/path/to/model_final.pth` with your trained checkpoint. The configuration lists COME15K-E, COME15K-H, DSIS, and SIP as evaluation datasets. Verify the script's argument handling and dataset paths before running.

The repository also contains `infer_rgbd_sis.py` for RGB-D inference. Consult its command-line arguments before supplying input RGB/depth paths and a checkpoint; no unverified inference flags are documented here.

## Experimental Results

The following values are **reported in the accompanying manuscript**; they are not newly reproduced measurements.

### Comparison with CalibNet

| Dataset | Method | AP ↑ | AP50 ↑ | AP70 ↑ |
|---|---|---:|---:|---:|
| COME15K-E | CalibNet | 58.0 | 75.8 | 65.6 |
| COME15K-E | **ADINet** | **58.8** | **76.5** | **66.8** |
| COME15K-H | CalibNet | 50.7 | 70.4 | 57.3 |
| COME15K-H | **ADINet** | **51.9** | **71.3** | **58.6** |
| DSIS | CalibNet | 69.3 | 87.8 | 81.6 |
| DSIS | **ADINet** | **70.2** | **88.5** | **82.7** |
| SIP | CalibNet | 72.1 | 86.6 | 82.9 |
| SIP | **ADINet** | **73.6** | **87.2** | **84.1** |

### Model Efficiency

| Method | Parameters (M) | FPS |
|---|---:|---:|
| CalibNet | 78.9 | 35.9 |
| **ADINet** | **78.6** | **44.5** |

The reported FPS values are from the paper's comparison; benchmark conditions should be matched before comparing them with measurements on other machines.

### Ablation Study on COME15K-H

| CMAE | ADIK | FADE | AP ↑ | AP50 ↑ | AP70 ↑ |
|:---:|:---:|:---:|---:|---:|---:|
| | | | 50.7 | 70.4 | 57.3 |
| ✓ | | | 51.1 | 70.7 | 57.5 |
| | ✓ | | 51.3 | 70.6 | 57.8 |
| | | ✓ | 51.2 | 70.9 | 58.0 |
| ✓ | ✓ | | 51.6 | 71.0 | 58.2 |
| ✓ | ✓ | ✓ | **51.9** | **71.3** | **58.6** |

The full configuration improves AP by 1.2 points and AP70 by 1.3 points over the reported baseline on COME15K-H.



## Contact

For questions, please open a [GitHub issue](https://github.com/heihei-450/ADINet/issues).

---

**Repository:** https://github.com/heihei-450/ADINet
