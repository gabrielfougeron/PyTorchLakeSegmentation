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


import torch
import warnings

from PIL import Image

def inBB(BB,x,y):
    return (x > BB[0]) and (y > BB[1]) and (x < BB[2]) and (y < BB[3])


input_image_folder = './input/images/'
input_poly_folder = './input/polygons/'

output_masks_folder = './output/Lakes_masks'
output_imgs_folder = './output/Lakes_png_images'

output_render_folder = './render'

image_filename_list = []
poly_filename_list = []


for file_path in os.listdir(input_poly_folder):
    file_path = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))
    
    if (file_ext == '.shp' ):
        
        image_filename = input_image_folder+file_root+'.TIF'
        
        if os.path.exists(image_filename):
            
            image_filename_list.append(image_filename)
            
            poly_filename = input_poly_folder+file_root+'.shp'
            poly_filename_list.append(poly_filename)
            
        else:
            print('')
            print(image_filename)
            raise RuntimeError("Matching image file not found")

for store_folder in [output_masks_folder,output_imgs_folder,output_render_folder]:
    if not(os.path.isdir(store_folder)):
        os.mkdir(store_folder)


n_img = len(image_filename_list)


for i_img in range(n_img):
    
    print('')
    print('----------------------------------------------')
    print('')
    print('Processing image '+str(i_img+1)+' of '+str(n_img))

    image_name = image_filename_list[i_img]
    lake_bou_name = poly_filename_list[i_img]
    
    print(image_name)
    print(lake_bou_name)

    # Open raster image
    with rio.open(image_name) as img_open:
        img = img_open.read()


        nx = img.shape[1]
        ny = img.shape[2]
        print('nx = ',nx)
        print('ny = ',ny)


        # Bounding box
        BB = plotting_extent(img_open)
        xmin,xmax,ymin,ymax = plotting_extent(img_open)
        print('xmin = ',xmin)
        print('xmax = ',xmax)
        print('ymin = ',ymin)
        print('ymax = ',ymax)


        #import lake boundaries
        lake_outlines = gpd.read_file(lake_bou_name)

        # Project lake boundaries to match Spot data
        lake_outline_match=lake_outlines.to_crs(img_open.crs)
        
        # ~ if ('Shape_Area' in  lake_outline_match.keys()) :
            # ~ print(lake_outline_match['Shape_Area'])
            # ~ print(lake_outline_match.area)
        

        npoly = lake_outline_match['geometry'].shape[0]
        print('npoly = ',npoly)

        if (npoly > 255):
            warnings.warn("Too many polygons in a single image", RuntimeWarning)

        polyBB = np.zeros((npoly,4))
        for ipoly in range(npoly):
            polyBB[ipoly,:] = lake_outline_match['geometry'][ipoly].bounds
            

        img_uint8 = ((img[0,:,:]/4)).astype(np.uint8)

        mask = np.zeros((nx,ny),dtype=np.uint8)
                
        for ipoly in range(npoly):

            MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)
    
            poly_raster = np.where(MA,1,0).astype(np.uint8).reshape((nx,ny))
            
            poly_collisions = np.where(MA,mask,0).astype(np.uint8).reshape((nx,ny))

            obj_ids = np.unique(poly_collisions)
            
            for ilake in obj_ids[1:]:
                
                print("Warning: Lake "+str(ilake)+" overlaps with lake "+str(ipoly+1))
        
            
            mask = np.where(MA,(ipoly+1),mask).astype(np.uint8).reshape((nx,ny))




print('')
print('Done !')
