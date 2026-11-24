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

## Highlights

- **Cross-Modal Attention Exchange (CMAE):** exchanges spatial attention responses between RGB and depth features to improve complementary fusion.
- **Adaptive Dynamic Interaction Kernel (ADIK):** combines input-dependent conditional convolution and cross-modal gating to generate adaptive instance-aware representations.
- **Frequency-Adaptive Depth Enhancement (FADE):** applies RGB-guided, channel-adaptive frequency filtering to depth features using real FFT and inverse real FFT.
- **Competitive performance:** evaluated on COME15K-E, COME15K-H, DSIS, and SIP.

## Architecture

The overall pipeline consists of:

1. **Dual-stream encoder:** separate RGB and depth feature extraction followed by deformable Transformer encoding.
2. **Mask Branch:** CMAE exchanges attention between modalities and constructs fused pixel-level mask representations.
3. **Kernel Branch:** FADE enhances depth representations and ADIK generates dynamic instance-aware kernels.
4. **Instance prediction:** instance kernels interact with mask features; training uses bipartite matching.

> **Figure placeholder:** Add the architecture figure from the paper to `assets/architecture.png`, then uncomment:
>
> `<!-- ![ADINet architecture](assets/architecture.png) -->`

### CMAE

CMAE compresses RGB/depth features, generates spatial attention for each modality, exchanges the attention maps across modalities, and fuses the modulated features with a residual connection. This allows semantic RGB cues and geometric depth cues to guide one another.

### ADIK

ADIK uses conditional convolution with multiple expert kernels, whose mixing coefficients depend on the input. Cross-modal complementary gating then supports instance-aware attention and kernel generation, improving adaptability to occlusion, scale variation, and complex boundaries.

### FADE

FADE derives channel-adaptive modulation from RGB features, transforms depth features into the frequency domain with RFFT, and combines a Gaussian low-pass basis with learnable channel-wise frequency responses. The filtered features are transformed back with IRFFT. This is intended to reduce the influence of noisy or incomplete depth measurements while retaining useful structure.

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

**Important:** Dataset download URLs and a fully verified directory layout are not provided here. Check the dataset registration implementation before preparing files; do not assume a generic COCO folder layout is accepted without conversion.

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

## Limitations

The paper identifies remaining challenges in detecting very small instances embedded within larger salient objects, suppressing structurally similar background regions, and preserving intricate object boundaries.

## Citation

If you use this repository, please cite the accompanying manuscript. A formal BibTeX entry with venue, year, and publication identifier should be added once those details are confirmed.

```bibtex
@misc{adinet,
  title  = {Adaptive Dynamic Interaction for RGB-D Salient Instance Segmentation},
  author = {Liu, Zhongmin and Zhang, Junrong and Hu, Wenjin},
  note   = {Manuscript; publication details to be updated}
}
```

## Acknowledgements

This project builds on the Detectron2 ecosystem and RGB-D salient instance segmentation research, including CalibNet. We thank the authors and maintainers of the relevant frameworks and datasets.

## Contact

For questions, please open a [GitHub issue](https://github.com/heihei-450/ADINet/issues).

---

**Repository:** https://github.com/heihei-450/ADINet
