# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import rasterio as rio
from rasterio.plot import plotting_extent

# from engine import train_one_epoch, evaluate,plot_model
from engine import *
import utils
import transforms as T

from tv_training_code import *

import shutil

# If memory issues then export this
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128




def main():
    # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')

    root_name = './lakes_data'
    dataset_test = LakesDataset(root_name, get_transform(train=False))
    
    eval_images_folder = "./eval_images"
    
    train_folder = "./trainings/5"
        
    tmp_folder = './tmp/'
    transfo_folder = './transformed/'
    transfo_masks_folder = transfo_folder+'Lakes_masks/'
    transfo_imgs_folder = transfo_folder+'Lakes_png_images/'

    the_folders = [transfo_masks_folder,transfo_imgs_folder]
    for fold in the_folders:
        if not(os.path.isdir(fold)):
            os.makedirs(fold)

    
    
    # input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/UniversiteParisSud_A.SEJOURNE_SO20004818-44-01_52862211010030255002B0/SCENE01/'
    # input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1989_07_16_Spot1/SCENE01/'
    input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/'
    


    thresh = 0.5
    
    # nx_out = 1024
    # ny_out = 1024
    
    nx_out = 2048
    ny_out = 2048
    
    # nx_out = 4096
    # ny_out = 4096
    
    nx_in = 1024
    ny_in = 1024
    
    input_img_list = []
    for file_path in os.listdir(input_image_folder):
        file_p = os.path.join(input_image_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_p))
        
        if (file_ext.lower() == '.tif' ):
            input_img_list.append(file_path)
            
            
    print(input_img_list)
            
    # input_img_list = [input_img_list[0]]
            
        
    mask = np.zeros((nx_out,ny_out),dtype=np.uint8)
    mask[0:(nx_out//2),0:(ny_out//2)]=1
    PIL_mask = Image.fromarray(mask)
    PIL_mask.save(transfo_masks_folder+"/mask_.png")
            
            
    n_input_img = len(input_img_list)

    model_list = []
    
    PlotExact=False
    # PlotExact=True
    

    for model_path in os.listdir(train_folder):
        model_path = os.path.join(train_folder, model_path)
        model_root, model_ext = os.path.splitext(os.path.basename(model_path))
    
        if (model_ext == '.pt' ):
    
            model_list.append([model_path,model_root])
    
    
    model_list_idx = [19,28,38,47,57,85]
    
    model_list = [model_list[i] for i in model_list_idx]
    
    # model_list = model_list[11:]
    # model_list = model_list[21:]
    # model_list = model_list[32:]
    
    for the_model in model_list:
        
        model_root = the_model[1]
        print(model_root)


    if PlotExact :
        

        plot_exact(data_loader_test,output_folder=tmp_folder)
        
        for file_path in os.listdir(tmp_folder):
            file_path = os.path.join(tmp_folder, file_path)
            file_root, file_ext = os.path.splitext(os.path.basename(file_path))

            store_folder = eval_images_folder+'/'+file_root+'/'
            if not(os.path.isdir(store_folder)):
                os.makedirs(store_folder)

            dst = store_folder+"exact.png"

            shutil.copyfile(file_path, dst)

        
    else:
        
        for the_model in model_list:
            
            model_path = the_model[0]
            model_root = the_model[1]
            
            print(model_path)
            

            model = torch.load(model_path)

            # move model to the right device
            model.to(device)
            
            i_img = 0
            
            for file_path in input_img_list:
                file_p = os.path.join(input_image_folder, file_path)
                file_root, file_ext = os.path.splitext(os.path.basename(file_p))
                
                i_img = i_img + 1
                print('')
                print('----------------------------------------------')
                print('')   
                print('Processing image '+str(i_img)+" of "+str(n_input_img))
                print(file_root)
                

                # Open raster image
                with rio.open(file_p) as img_open:
                    img = img_open.read()
                    print("image loaded")
                    # img_uint8 = ((img[0,:,:]//4)).astype(np.uint8)
                    # img_uint8 = ((img[0,:,:]//2)).astype(np.uint8)
                    # img_uint8 = ((img[0,:,:]*2.5)).astype(np.uint8)
                    img_uint8 = ((img[0,:,:]*1.8)).astype(np.uint8)
                    
                    del img
                    
                    
                    nxtot = img_uint8.shape[0]
                    nytot = img_uint8.shape[1]
                    print('nxtot = ',nxtot)
                    print('nytot = ',nytot)
                        
                    BB = plotting_extent(img_open)
                    xmin,xmax,ymin,ymax = plotting_extent(img_open)
                    # xmin,ymin,xmax,ymax = img_open.bounds
                    print('xmin = ',xmin)
                    print('xmax = ',xmax)
                    print('ymin = ',ymin)
                    print('ymax = ',ymax)

                    
                    # xstart_list = [0,nx_out//2]
                    # ystart_list = [0,ny_out//2]
                    
                    xstart_list = [0]
                    ystart_list = [0]
                    
                    istartmin = 0

                    
                    
                    for istart in range(istartmin,len(xstart_list)):
                        
                        xstart = xstart_list[istart]
                        ystart = ystart_list[istart]
                            
                        ixmin = 0
                        iymin = 0

                        ixmax = (nxtot-xstart)//nx_out
                        iymax = (nytot-ystart)//ny_out
                    
                        for ix in range(ixmin,ixmax):
                            for iy in range(iymin,iymax):
                            
                            
                                    
                                # ixlist = [13,14,15]
                                # iylist = [12,13,14]
                                    
                                # ixlist = [6,7,8]
                                # iylist = [1,2,3]

                                # for ix in ixlist:
                                    # for iy in iylist:
                                
                                
                                
                                # print(istart,ix+1,ixmax,iy+1,iymax)
                                print(istart,ix+1,iy+1)
                        
                    
                                img_uint8_small = img_uint8[xstart+ix*nx_out:xstart+(ix+1)*nx_out,ystart+iy*ny_out:ystart+(iy+1)*ny_out]
                                
                                PIL_img = Image.fromarray(img_uint8_small)
                                PIL_img = PIL_img.resize(size=(nx_in,ny_in),resample=Image.BICUBIC)
                                PIL_img.save(transfo_imgs_folder+"/img_.png")
                                
                                
                                
                                
                                # PIL_img = Image.fromarray(img_uint8_small)
                                # PIL_img.save(transfo_imgs_folder+"/img_.png")
                                
                                dataset_test = LakesDataset(transfo_folder, get_transform(train=False))
                                
                                data_loader_test = torch.utils.data.DataLoader(
                                    dataset_test, batch_size=1, shuffle=False, num_workers=1,
                                    collate_fn=utils.collate_fn)

                                output_img_filename = tmp_folder+file_root+'_'+str(istart)+'_'+str(ix).zfill(2)+'_'+str(iy).zfill(2)+'.png'
                                
                                
                                all_masks = get_one_mask_and_plot(model, data_loader_test, device=device,image_output_filename=output_img_filename,thresh=thresh)
                                
                                
            del model
            
            for file_path in os.listdir(tmp_folder):
                file_path = os.path.join(tmp_folder, file_path)
                file_root, file_ext = os.path.splitext(os.path.basename(file_path))

                store_folder = eval_images_folder+'/'+file_root+'/'
                if not(os.path.isdir(store_folder)):
                    os.makedirs(store_folder)

                dst = store_folder+model_root+".png"

                shutil.copyfile(file_path, dst)


    print("That's it!")
    
if __name__ == "__main__":
    main()
