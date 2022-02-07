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
import colorsys
import random

import shapely
import rasterstats


import torch
import warnings

from PIL import Image

def inBB(BB,x,y):
    return (x > BB[0]) and (y > BB[1]) and (x < BB[2]) and (y < BB[3])


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


input_image_folder = './input/images/'
input_poly_folder = './input/polygons/'
# ~ input_poly_folder = './input/polygons_merged/'

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
            

        # ~ print(img.dtype)
        # ~ print(np.amax(img))
        # ~ print(np.amin(img))
        # ~ mask = (((img[0,:,:] - np.amin(img))/(np.amax(img) - np.amin(img))) * 255).astype(np.uint8)

        # ~ img_uint8 = (img[0,:,:]/16).astype(np.uint8)
        # ~ img_uint8 = (np.minimulmmin(img[0,:,:]/4,255)).astype(np.uint8)
        img_uint8 = ((img[0,:,:]/4)).astype(np.uint8)



        mask = np.zeros((nx,ny),dtype=np.uint8)
                
        for ipoly in range(npoly):

            MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)

            poly_raster = np.where(MA,1,0).astype(np.uint8).reshape((nx,ny))
            
            overlap = np.sum(poly_raster*mask)
            
            if (overlap != 0):
                # ~ print('XXXXXXXXXXXXXXXXXXXXXXXXXX')
                # ~ print(overlap)
                raise RuntimeWarning("POLYGON OVERLAP involving polygon "+str(ipoly))
            
            
            # ~ print(poly_raster)
            # ~ print((ipoly+1),np.sum(poly_raster))
            
            mask += (ipoly+1)*poly_raster

            
        img_out_filename = output_imgs_folder+"/img_"+str(i_img)+".png"
        msk_out_filename = output_masks_folder+"/mask_"+str(i_img)+".png"
          
        print("Saving in "+img_out_filename)
        
        PIL_img = Image.fromarray(img_uint8)
        PIL_img.save(img_out_filename)

        PIL_mask = Image.fromarray(mask)
        PIL_mask.save(msk_out_filename)



    colors = random_colors(npoly)


    fig = plt.figure(figsize=(10,10))
    ax=plt.gca()

    ep.plot_bands(img,ax=ax, cbar=False,extent=BB)
    # ~ lake_outlines.plot(ax=ax) 
    lake_outline_match.plot(ax=ax,color=colors,alpha=0.4) 

    plt.tight_layout()
    # ~ plt.show()
    plt.savefig(output_render_folder+"/out_"+str(i_img)+".png")
    plt.close()



print('')
print('Done !')


