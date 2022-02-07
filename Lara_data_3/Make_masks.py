# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:34:54 2021

@author: Lara
"""
import os
import math
import time
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

import gc

import torch
import warnings

from scipy import ndimage

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




# init_image = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF'
# init_lake_bou = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03_North_clipB.shp'
# img_name_id = '2010_10_03_N'
# brightness_coeff_baseline = 1.


# init_image = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF'
# init_lake_bou = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_S/2010_10_03_lakes_clipped.shp'
# img_name_id = '2010_10_03_S'
# brightness_coeff_baseline = 1.


init_image = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2012_09_25/SCENE01/2012_09_25_Spot5.TIF'
init_lake_bou = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2012_09_25/2012_09_25B.shp'
img_name_id = '2012_09_25'
brightness_coeff_baseline = 1.



# init_image = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2016_Spot7/Spot7_Syrdakh_BW.tif'
# init_lake_bou = '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2016_Spot7/2016_Spot7.shp'
# img_name_id = '2016_Spot7'
# brightness_coeff_baseline = 0.25





output_masks_folder = './output/Lakes_masks'
output_imgs_folder = './output/Lakes_png_images'

output_render_folder = './render'


for store_folder in [output_masks_folder,output_imgs_folder,output_render_folder]:
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

# nx_out = 2048
# ny_out = 2048

nx_out = 1024
ny_out = 1024

scale_min = 0.5
scale_max = 3.

n_img_output = 2000


print('Image path :')
print(init_image)

print('Shapefile path :')
print(init_lake_bou)
print('')

with rio.open(init_image) as img_open:

    img = img_open.read()
    # img_uint8 = ((img[0,:,:]/4)).astype(np.uint8)
    img_uint8 = ((img[0,:,:]*brightness_coeff_baseline)).astype(np.uint8)

    nx_img = img.shape[1]
    ny_img = img.shape[2]
    print('nx = ',nx_img)
    print('ny = ',ny_img)

    del img

    # Bounding box
    BB = plotting_extent(img_open)
    xmin,xmax,ymin,ymax = plotting_extent(img_open)
    print('xmin = ',xmin)
    print('xmax = ',xmax)
    print('ymin = ',ymin)
    print('ymax = ',ymax)

    lake_outlines = gpd.read_file(init_lake_bou)
    lake_outline_match=lake_outlines.to_crs(img_open.crs)

    npoly = lake_outline_match['geometry'].shape[0]
    print('npoly = ',npoly)

    poly_img = np.zeros((nx_img,ny_img),dtype=np.uint16)
    
    for ipoly in range(npoly):

        print("ipoly = ",ipoly,' / ',npoly)

        # MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)
        MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=False)

        overlap_polys = np.ma.masked_array(poly_img,mask=MA)            
        overlap = (overlap_polys.max() != 0)

        if (overlap):
            # print('XXXXXXXXXXXXXXXXXXXXXXXXXX')
            # print(overlap)
            raise RuntimeWarning("POLYGON OVERLAP involving polygon "+str(ipoly))
        
        poly_img = np.where(MA,poly_img,(ipoly+1))  

    del MA
    del T
    del _
    del overlap_polys
    
    gc.collect()
    
    

safe_scale = np.sqrt(2.)

safe_min = (2. - np.sqrt(2.))/4
safe_max = (2. + np.sqrt(2.))/4

for i_out in range(n_img_output):
    
    print(i_out)
    
    the_scale = scale_min + (scale_max-scale_min)*random.random()
    
    dx = int(nx_out * the_scale * safe_scale)
    dy = int(ny_out * the_scale * safe_scale)
    
    ixmin = random.randrange(0,nx_img-dx)
    iymin = random.randrange(0,ny_img-dy)

    ixmax = ixmin+dx
    iymax = iymin+dy
    
    sub_img = np.copy(img_uint8[ixmin:ixmax,iymin:iymax])
    sub_poly_img = np.copy(poly_img[ixmin:ixmax,iymin:iymax])
    
    rot_angle = 360*random.random()
    
    brightness_coeff_min = 0.7
    brightness_coeff_max = 1.5
    brightness_coeff = brightness_coeff_min + (brightness_coeff_max - brightness_coeff_min) * random.random()
    
    sub_img_rot = (ndimage.rotate(sub_img,rot_angle,reshape=False,order=3) * brightness_coeff).astype(np.uint8)
    sub_poly_img_rot = ndimage.rotate(sub_poly_img,rot_angle,reshape=False,order=0)
    
    # print(dx,dy)
    # print(sub_poly_img_rot.shape)
    
    ixmin_rot = int(safe_min * dx)
    iymin_rot = int(safe_min * dy)

    ixmax_rot = int(safe_max * dx)
    iymax_rot = int(safe_max * dy)
    
    sub_img_rot = sub_img_rot[ixmin_rot:ixmax_rot,iymin_rot:iymax_rot]
    sub_poly_img_rot = sub_poly_img_rot[ixmin_rot:ixmax_rot,iymin_rot:iymax_rot]
    
    PIL_img = Image.fromarray(sub_img_rot)
    PIL_img = PIL_img.resize(size=(nx_out,ny_out),resample=Image.BICUBIC)
    sub_img_rot = np.array(PIL_img)
    
    PIL_mask = Image.fromarray(sub_poly_img_rot)
    PIL_mask =  PIL_mask.resize(size=(nx_out,ny_out),resample=Image.NEAREST)
    sub_poly_img_rot = np.array(PIL_mask)
    
    obj_ids, obj_counts = np.unique(sub_poly_img_rot, return_counts=True)
        
    pxl_thresh = 5
    n_obj = obj_ids.shape[0]
    
    n_obj_thr = 0
    
    for i in range(n_obj):
        if (obj_counts[i] >= pxl_thresh):
            n_obj_thr += 1
    
    obj_ids_thr = np.zeros((n_obj_thr),dtype=obj_ids.dtype)
    obj_counts_thr = np.zeros((n_obj_thr),dtype=obj_counts.dtype)
    
    n_obj_thr = 0
    
    for i in range(n_obj):
        if (obj_counts[i] >= pxl_thresh):
    
            obj_ids_thr[n_obj_thr] = obj_ids[i]
            obj_counts_thr[n_obj_thr] = obj_counts[i]

            n_obj_thr += 1

    # print(i_img,ix,iy)
    print("npoly = ",obj_counts_thr.size)
    # print('')
    
    if (obj_ids_thr.size > 1):
            
        img_out_filename = output_imgs_folder+"/img_"+img_name_id+"_"+str(i_out).zfill(5)+".png"
        msk_out_filename = output_masks_folder+"/mask_"+img_name_id+"_"+str(i_out).zfill(5)+".png"

          
        print("Saving in "+img_out_filename)
        
        PIL_img = Image.fromarray(sub_img_rot)
        PIL_img.save(img_out_filename)

        obj_ids_thr = np.sort(obj_ids_thr)
        sub_poly_img_rot = np.searchsorted(obj_ids_thr,sub_poly_img_rot).astype(np.uint16)

        PIL_mask = Image.fromarray(sub_poly_img_rot)
        PIL_mask.save(msk_out_filename)
        
    print("")



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            


print('')
print('Done !')


