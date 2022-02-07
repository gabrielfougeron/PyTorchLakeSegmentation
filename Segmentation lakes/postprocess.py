# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ~ from engine import train_one_epoch, evaluate,plot_model
from engine import *
import utils
import transforms as T

from tv_training_code import *


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    # ~ device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')

    root_name = './lakes_data'
    dataset_test = LakesDataset(root_name, get_transform(train=False))
    
    # ~ # use our dataset and defined transformations
    # ~ root_name = './lakes_data'
    # ~ dataset_test = LakesDataset_test(root_name, get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # ~ model = torch.load("./MyTraining_10.pt")
    # ~ model = torch.load("./MyTraining_final.pt")
    model = torch.load("./b.pt")

    # move model to the right device
    model.to(device)


    plot_model(model, data_loader_test, device=device)
    # ~ plot_exact(data_loader_test)


    print("That's it!")
    
if __name__ == "__main__":
    main()
