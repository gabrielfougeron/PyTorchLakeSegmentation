# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from engine import train_one_epoch, evaluate,plot_model
from engine import *
import utils
import transforms as T

from tv_training_code import *

import shutil

import gc





def main():
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')

    root_name = './lakes_data'
    dataset_test = LakesDataset(root_name, get_transform(train=False))
    
    eval_images_folder = "./eval_images"
    
    train_folder = "./trainings/3"
    
    eval_indices = np.loadtxt(train_folder+"/eval_set.txt", dtype=int)
    print(eval_indices)
    dataset_test = torch.utils.data.Subset(dataset_test, eval_indices)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model_list = []
    

    for model_path in os.listdir(train_folder):
        model_path = os.path.join(train_folder, model_path)
        model_root, model_ext = os.path.splitext(os.path.basename(model_path))
    
        if (model_ext == '.pt' ):
    
            model_list.append([model_path,model_root])
    
    
    model_list = model_list[:10]
    # model_list = model_list[11:]
    # model_list = model_list[21:]
    # model_list = model_list[32:]
    
    for the_model in model_list:
        
        model_root = the_model[1]
        print(model_root)

    
    for the_model in model_list:
        
        model_path = the_model[0]
        model_root = the_model[1]
        

        print(model_path)
        
        
        model = torch.load(model_path)

        # move model to the right device
        model.to(device)
        
        
        tmp_folder = './tmp/'
        
        plot_model(model, data_loader_test, device=device,output_folder=tmp_folder)
        
        del model
        
        for file_path in os.listdir(tmp_folder):
            file_path = os.path.join(tmp_folder, file_path)
            file_root, file_ext = os.path.splitext(os.path.basename(file_path))

            store_folder = eval_images_folder+'/'+file_root+'/'
            if not(os.path.isdir(store_folder)):
                os.makedirs(store_folder)

            dst = store_folder+model_root+".png"

            shutil.copyfile(file_path, dst)
        
        gc.collect()


    print("That's it!")
    
if __name__ == "__main__":
    main()
