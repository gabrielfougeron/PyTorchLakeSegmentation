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

from PIL import Image,ImageFilter

def inBB(BB,x,y):
    return (x > BB[0]) and (y > BB[1]) and (x < BB[2]) and (y < BB[3])


init_image_list = [
'/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
'/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
'/mnt/c/GeoData/training_gab_01_23_2022/2012_09_25/SCENE01/2012_09_25_Spot5.TIF',
'/mnt/c/GeoData/training_gab_01_23_2022/2016_Spot7/Spot7_Syrdakh_BW.tif',
]

init_lake_bou_list = [
'/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03_North_clipB.shp',
'/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/2010_10_03_lakes_clipped.shp',
'/mnt/c/GeoData/training_gab_01_23_2022/2012_09_25/2012_09_25_lakes.shp',
'/mnt/c/GeoData/training_gab_01_23_2022/2016_Spot7/2016_Spot7.shp',
]

img_name_id_list = [
'2010_10_03_N',
'2010_10_03_S',
'2012_09_25',
'2016_Spot7',
]

n_img_output_list = [
3000,
3000,
8000,
8000,
]

mod_img_lara_fix_list = [
True,
True,
False,
False,
]

Conversion_factor_to_uint8_list = [
1.0,
1.0,
1.0,
0.25,
]

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

scale_min = 1.7
scale_max = 2.5

target_mean_min = 100
target_mean_max = 160

target_stddev_min = 10
target_stddev_max = 60

Local_contrast = True
# Local_contrast = False


BLUR_proba = 0.1
SMOOTH_proba = 0.1
SMOOTH_MORE_proba = 0.1



# for i_img in range(len(init_image_list)):
for i_img in [3]:
    
    init_image = init_image_list[i_img]
    init_lake_bou = init_lake_bou_list[i_img]
    img_name_id = img_name_id_list[i_img]
    n_img_output = n_img_output_list[i_img]
    
    print('Image path :')
    print(init_image)

    print('Shapefile path :')
    print(init_lake_bou)
    print('')

    with rio.open(init_image) as img_open:
        
        img = img_open.read()
            
        nx_img = img.shape[1]
        ny_img = img.shape[2]
        print('nx = ',nx_img)
        print('ny = ',ny_img)
                
        BB = plotting_extent(img_open)
        xmin,xmax,ymin,ymax = plotting_extent(img_open)
        # xmin,ymin,xmax,ymax = img_open.bounds
        print('xmin = ',xmin)
        print('xmax = ',xmax)
        print('ymin = ',ymin)
        print('ymax = ',ymax)
        
        if mod_img_lara_fix_list[i_img]:
            # fix for cropped images
            print(img.dtype)
            img = np.where(img[0,:,:] == 256 ,np.uint16(0),img)

        if Local_contrast:
            
            img_uint8 = (img[0,:,:]*Conversion_factor_to_uint8_list[i_img]).astype(np.uint8)
            
            del img
            
        else:
            
            target_mean = 120
            target_stddev = 40
                
            vals, count = np.unique(img , return_counts=True)
            
            vals = vals[1:]
            count = count[1:]
            
            print(vals)
            
            mean = np.sum(vals*count)/np.sum(count)
            stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
          
            img_new = ((img.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
            
            img_new = np.where(img_new > 255.,255.,img_new)
            img_new = np.where(img_new < 0.,0.,img_new)
            img_new = img_new.astype(np.uint8)

            
            img_uint8 = np.where(img[0,:,:] == 0 ,np.uint8(0),img_new[0,:,:])

            del img
            del img_new
        
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

        if Local_contrast:

            target_mean = target_mean_min + (target_mean_max - target_mean_min) * random.random()
            target_stddev = target_stddev_min + (target_stddev_max - target_stddev_min) * random.random()

            vals, count = np.unique(sub_img , return_counts=True)
            
            vals = vals[1:]
            count = count[1:]
            
            mean = np.sum(vals*count)/np.sum(count)
            stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
          
            sub_img_new = ((sub_img.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
            
            sub_img_new = np.where(sub_img_new > 255.,255.,sub_img_new)
            sub_img_new = np.where(sub_img_new < 0.,0.,sub_img_new)
            
            sub_img = np.where(sub_img == 0 ,0.,sub_img_new)


        
        sub_img_rot = ndimage.rotate(sub_img,rot_angle,reshape=False,order=3).astype(np.uint8)
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
        
                
        if (random.random() < BLUR_proba):
            PIL_img = PIL_img.filter(ImageFilter.BLUR)
        if (random.random() < SMOOTH_proba):
            PIL_img = PIL_img.filter(ImageFilter.SMOOTH)
        if (random.random() < SMOOTH_MORE_proba):
            PIL_img = PIL_img.filter(ImageFilter.SMOOTH_MORE)
            



        
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

        print("npoly = ",obj_counts_thr.size)
        # print('')
        
        if (obj_ids_thr.size > 1):
                
            img_out_filename = output_imgs_folder+"/img_"+img_name_id+"_"+str(i_out).zfill(5)+".png"
            msk_out_filename = output_masks_folder+"/mask_"+img_name_id+"_"+str(i_out).zfill(5)+".png"

              
            print("Saving in "+img_out_filename)
            
            PIL_img.save(img_out_filename)

            obj_ids_thr = np.sort(obj_ids_thr)
            sub_poly_img_rot = np.searchsorted(obj_ids_thr,sub_poly_img_rot).astype(np.uint16)

            PIL_mask = Image.fromarray(sub_poly_img_rot)
            PIL_mask.save(msk_out_filename)
            
        print("")



print('')
print('Done !')


