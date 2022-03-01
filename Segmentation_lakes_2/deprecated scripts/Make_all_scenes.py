# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:34:54 2021

@author: Lara
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import rasterio as rio
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import fiona

import shapely
import rasterstats


import warnings

import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import *
import utils
import transforms as T

from tv_training_code import *

device = torch.device('cuda')

input_image_folder = './input/all_images/'

transfo_folder = './transformed/'
transfo_masks_folder = transfo_folder+'Lakes_masks/'
transfo_imgs_folder = transfo_folder+'Lakes_png_images/'

output_poly_folder = './polygons/polygons/'
output_imgs_folder = './polygons/images/'

for store_folder in [transfo_folder,transfo_masks_folder,transfo_imgs_folder,output_imgs_folder,output_poly_folder]:
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)
        




# ~ model = torch.load("./MyTraining_10.pt")
# ~ model = torch.load("./MyTraining_final.pt")
model = torch.load("./trainings/2/MyTraining_final.pt")
model.to(device)

input_img_list = []
for file_path in os.listdir(input_image_folder):
    file_p = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_p))
    
    if (file_ext == '.TIF' ):
        input_img_list.append(file_path)
        
        

# ~ input_img_list = input_img_list[130:]
# ~ input_img_list = input_img_list[:10]

        

n_input_img = len(input_img_list)

i_img = 0

for file_path in input_img_list:
    file_p = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_p))
    
    poly_filename = output_poly_folder+file_root+'.shp'
        
    i_img = i_img + 1
    print('')
    print('----------------------------------------------')
    print('')   
    print('Processing image '+str(i_img)+" of "+str(n_input_img))
    print(file_root)
    

    # Open raster image
    with rio.open(file_p) as img_open:
        img = img_open.read()
            
        nx = img.shape[1]
        ny = img.shape[2]
        print('nx = ',nx)
        print('ny = ',ny)
            
        BB = plotting_extent(img_open)
        xmin,xmax,ymin,ymax = plotting_extent(img_open)
        print('xmin = ',xmin)
        print('xmax = ',xmax)
        print('ymin = ',ymin)
        print('ymax = ',ymax)
        
        img_uint8 = ((img[0,:,:]/4)).astype(np.uint8)
        PIL_img = Image.fromarray(img_uint8)
        PIL_img.save(transfo_imgs_folder+"/tmp_img.png")
        
        mask = np.zeros((nx,ny),dtype=np.uint8)
        mask[0:(nx//2),0:(ny//2)]=1
        PIL_mask = Image.fromarray(mask)
        PIL_mask.save(transfo_masks_folder+"/tmp_mask.png")

        
        dataset_test = LakesDataset(transfo_folder, get_transform(train=False))
        
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        output_img_filename = output_imgs_folder+file_root+'.png'
        output_poly_filename = output_poly_folder+file_root+'.shp'
        
        thresh = 0.5
        
        all_masks = get_one_mask_and_plot(model, data_loader_test, device=device,image_output_filename=output_img_filename,thresh=thresh)
        # ~ all_masks = get_one_mask_no_plot(model, data_loader_test, device=device,thresh=thresh)
        
        npoly = all_masks.shape[0]
        print('npoly = ',npoly)
        
        polyval = []
        geometry = []
        
        for ipoly in range(npoly):
            
            mask = torch.where(all_masks[ipoly,:,:] > thresh,1,0).reshape((nx,ny)).detach().cpu().numpy().astype(np.uint8)
            
            the_polys = rio.features.shapes(mask, transform=img_open.transform)

            for shapedict, value in the_polys:
                if (value == 1):
                    
                    polyval.append(str(ipoly))
                    geometry.append(shapely.geometry.shape(shapedict))
            

        
        
        
        
            
            
        
        # ~ max_mask = torch.amax(all_masks, axis=0).reshape((nx,ny))
        # ~ argmax_mask = torch.argmax(all_masks, axis=0).reshape((nx,ny)) +1
        # ~ mask = torch.where(max_mask > thresh,argmax_mask,0).detach().cpu().numpy().astype(np.uint8)
        # ~ all_polys = rio.features.shapes(mask, transform=img_open.transform)

        # ~ # read the shapes as separate lists
        # ~ polyval = []
        # ~ geometry = []
        # ~ for shapedict, value in all_polys:
            # ~ if (value > 0.5):
                
                # ~ polyval.append(str(int(value))) # ???
                # ~ geometry.append(shapely.geometry.shape(shapedict))
        
        # ~ print('npoly = ',len(polyval))
        
        # build the gdf object over the two lists
        gdf = gpd.GeoDataFrame(
            {'Id': polyval, 'geometry': geometry },
            crs=img_open.crs
        )
        
        gdf.to_crs({'proj':'cea'},inplace=True) 
        
        gdf['Shape_Area'] = gdf.area
        
        gdf.to_crs(img_open.crs,inplace=True) 
                    
        gdf.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
        

                        
                            
            
            
            
            
            

        

print('')
print('Done !')
