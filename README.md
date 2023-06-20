<div align="center">

# An Autonomous X-ray Image Acquisition and Interpretation System for Assisting Percutaneous Pelvic Fracture Fixation

![Overview](/images/overview.png)

</div>
<div align="left">

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
