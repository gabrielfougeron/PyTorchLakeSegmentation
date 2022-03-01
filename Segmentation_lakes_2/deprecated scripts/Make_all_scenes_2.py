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
# ~ input_image_folder = './input/DZB1216_1980_SPLIT/'

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
# ~ model = torch.load("./trainings/2/MyTraining_final.pt")
model = torch.load("./trainings/3/MyTraining_068.pt")
model.to(device)

input_img_list = []
for file_path in os.listdir(input_image_folder):
    file_p = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_p))
    
    if (file_ext == '.TIF' ):
        input_img_list.append(file_path)
        
        

# ~ input_img_list = input_img_list[155:160]
input_img_list = input_img_list[130:]
# ~ input_img_list = input_img_list[100:10]
# ~ input_img_list = input_img_list[8:10]

        

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
        
        npoly = all_masks.shape[0]
        print('npoly = ',npoly)
        
        mask_no_col = np.zeros((nx,ny),dtype=np.uint8)
        
        npoly_real = 0
        
        eq_classes = []
        # ~ poly_to_eq_class = []
        # ~ n_eq_class = 0
        
        for ipoly in range(npoly):
            
            MA = torch.where(all_masks[ipoly,:,:] > thresh,1,0).reshape((nx,ny)).detach().cpu().numpy().astype(np.uint8)
            
            poly_collisions = np.where(MA,mask_no_col,0).astype(np.uint8).reshape((nx,ny))
            
            partial_class = set(np.unique(poly_collisions))
            partial_class.remove(0)
            partial_class.add(ipoly+1)
            
            
            for the_class in eq_classes:
                
                if partial_class.intersection(the_class):
                    
                    partial_class.update(the_class)
            
            eq_classes = [ the_class for the_class in eq_classes if not(partial_class.intersection(the_class))]
            eq_classes.append(partial_class)
            
            
            mask_no_col = np.where(MA,(ipoly+1),mask_no_col).astype(np.uint8).reshape((nx,ny))


        class_mask = np.zeros((nx,ny),dtype=np.uint8)
        
        for iclass in range(len(eq_classes)):
            
            for jpoly in eq_classes[iclass]:
            
                to_add = np.where(mask_no_col == jpoly ,(iclass+1),0).astype(np.uint8).reshape((nx,ny))
                
                overlap = np.sum(to_add*class_mask)
                
                if (overlap):
                    raise ValueError("There was an error in merging overlapping polygons")
                
                class_mask += to_add
        
        
        print('npoly without collisions = ',len(eq_classes))    

        polyval = []
        geometry = []
        
        the_polys = rio.features.shapes(class_mask, transform=img_open.transform)

        for shapedict, value in the_polys:
            if (value != 0):
                
                polyval.append(str(value-1)) # ici ça commence à zéro, désolé
                geometry.append(shapely.geometry.shape(shapedict))
        
        
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
