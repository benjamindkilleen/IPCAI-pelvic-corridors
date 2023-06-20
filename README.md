<div align="center">

# An Autonomous X-ray Image Acquisition and Interpretation System for Assisting Percutaneous Pelvic Fracture Fixation

![Overview](/images/overview.png)

</div>
<div align="left">

## Summary

Code for our paper, [An Autonomous X-ray Image Acquisition and Interpretation System for Assisting Percutaneous Pelvic Fracture Fixation](https://link.springer.com/article/10.1007/s11548-023-02941-y), which was presented at IPCAI 2023 in Munich, Germany.

Percutaneous fracture fixation involves multiple X-ray acquisitions to determine adequate tool trajectories in bony anatomy. In order to reduce time spent adjusting the X-ray imager’s gantry, avoid excess acquisitions, and anticipate inadequate trajectories before penetrating bone, we propose an autonomous system for intra-operative feedback that combines robotic X-ray imaging and machine learning for automated image acquisition and interpretation, respectively. Our approach reconstructs an appropriate trajectory in a two-image sequence, where the optimal second viewpoint is determined based on analysis of the first image. A deep neural network is responsible for detecting the tool and corridor, here a K-wire and the superior pubic ramus, respectively, in these radiographs. The reconstructed corridor and K-wire pose are compared to determine likelihood of cortical breach, and both are visualized for the clinician in a mixed reality environment that is spatially registered to the patient and delivered by an optical see-through head-mounted display.

Users of this repository may be primarily interested in how we generated our dataset and performed data augmentation. These methods are in the `ctpelvic1k.py` dataset file.

## Installation

If desired, place datasets in `./datasets/`. By default, they will be downloaded there when the code runs, but you may already have them on your system. For example, soft-link the CTPelvic1K directory with

```bash
mkdir datasets
cd datasets
ln -s /PATH/TO/CTPelvic1K .
cd ..
```

Then install the conda environment with

```bash
conda env create -f environment.yaml
conda activate cortical-breach-detection
```

## Usage

Activate the environment.

```bash
conda activate cortical-breach-detection
```

Run dataset generation/training:

```bash
python main.py experiment=train checkpoint=/PATH/TO/last.ckpt
```

Other experiments include

- `test`: test the model on the test set
- `triangulate`: test the full triangulation on the test set, simulating new images (requires two GPUs)
- `deepfluoro`: test the model on cadaver data from DeepFluoro
- `server`: run the server for communicating with the HMD.

See `conf/config.yaml` for more details.

</div>
