import os
import torch


def makeDirectory(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def set_gpu(gpus):
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        torch.cuda.set_device(int(gpus))