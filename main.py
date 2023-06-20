#!/usr/bin/env python3
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
from time import time
from tkinter import Frame


import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from skimage.transform import resize

# Nosort
import deepdrr
import cortical_breach_detection
from cortical_breach_detection.datasets.ctpelvic1k import CTPelvic1KDataModule
from cortical_breach_detection.loopx_app import start_loopx_window
from cortical_breach_detection.procs import CorridorServer, HololensServer
from cortical_breach_detection.models.unet import UNet
from cortical_breach_detection.robdata import triangulate_robdata
from deepdrr import geo
from deepdrr.utils import image_utils
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.progress import track
from rich.traceback import install

install(show_locals=False)

os.environ["HYDRA_FULL_ERROR"] = "1"

# Use agg backend for plotting when no graphical display available.
mpl.use("agg")

plt.rcParams["font.family"] = "Times New Roman"

log = logging.getLogger("cortical_breach_detection")


@cortical_breach_detection.register_experiment
def train(cfg):
    pl.seed_everything(cfg.seed)
    dm = CTPelvic1KDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    dm.prepare_data()
    dm.setup(stage="fit")
    model = UNet(**cfg.model)
    trainer = pl.Trainer(resume_from_checkpoint=cfg.checkpoint, **cfg.trainer)
    trainer.fit(model, datamodule=dm)

    # Appears to delete the old checkpoints
    # trainer.fit(model, datamodule=dm, ckpt_path=cfg.checkpoint)

    dm.setup(stage="test")
    trainer.test(model, datamodule=dm)


@cortical_breach_detection.register_experiment
def test(cfg):
    dm = CTPelvic1KDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    dm.prepare_data()
    model = UNet.load_from_checkpoint(cfg.checkpoint, **cfg.model)
    trainer = pl.Trainer(**cfg.trainer)
    dm.setup(stage="test")
    trainer.test(model, datamodule=dm)


@cortical_breach_detection.register_experiment
def deepfluoro(cfg):
    model = UNet.load_from_checkpoint(cfg.checkpoint, **cfg.model)
    triangulate_robdata(model, **cfg.data)

@cortical_breach_detection.register_experiment
def triangulate(cfg):
    """Run the in-silico triangulation experiment, which includes viewpoint planning."""
    dm = CTPelvic1KDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    dm.prepare_data()
    dm.setup(stage="test")
    test_set = dm.test_set
    model = UNet.load_from_checkpoint(cfg.checkpoint, **cfg.model)

    # Send pytorch model to GPU not in use by pycuda. Pycuda will, by default, take the first gpu.
    # If they are on the same gpu, DeepDRR will throw confusing errors.
    # Can reverse which physical gpu is used by setting the env variable CUDA_VISIBLE_DEVICES=1,0
    model.to(torch.device("cuda:1"))

    # Params from config
    test_set.triangulation(model)


@cortical_breach_detection.register_experiment
def server(cfg):
    """Run the server."""
    OmegaConf.resolve(cfg)

    mp.set_start_method("spawn", force=True)
    corridor_server_queue = mp.Queue(20)
    loopx_app_queue = mp.Queue(10)
    pose_publisher_queue = mp.Queue(10)
    image_publisher_queue = mp.Queue(10)
    hololens_server_queue = mp.Queue(5)

    # frame_listener = FrameListener(corridor_server_queue, **cfg.frame_listener)
    # frame_listener.start()

    hololens_server = HololensServer(
        hololens_server_queue, corridor_server_queue, **cfg.hololens_server
    )
    hololens_server.start()

    corridor_server = CorridorServer(
        corridor_server_queue,
        loopx_app_queue,
        pose_publisher_queue,
        image_publisher_queue,
        hololens_server_queue,
        **cfg.corridor_server,
    )
    corridor_server.start()

    start_loopx_window(loopx_app_queue, corridor_server_queue)

    # frame_listener.terminate()
    hololens_server.terminate()
    corridor_server.terminate()


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    cortical_breach_detection.run(cfg)


if __name__ == "__main__":
    main()
