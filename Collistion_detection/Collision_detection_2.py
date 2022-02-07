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
import gc

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


print('Image path :')
print(init_image)

print('Shapefile path :')
print(init_lake_bou)
print('')

were_there_overlaps = False

with rio.open(init_image) as img_open:

    img = img_open.read()
    img_uint8 = ((img[0,:,:])).astype(np.uint8)

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

        # ~ MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)
        MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=False)

        overlap_polys = np.ma.masked_array(poly_img,mask=MA)            
        overlap = (overlap_polys.max() != 0)

        if (overlap):
            # ~ print('XXXXXXXXXXXXXXXXXXXXXXXXXX')
            # ~ print(overlap)
            # raise RuntimeWarning("POLYGON OVERLAP involving polygon "+str(ipoly))
            
            the_collisions = np.unique(overlap_polys)
            print("POLYGON OVERLAP involving polygon "+str(ipoly)+" and something in ",the_collisions[1:]-1)
            
            were_there_overlaps = True
            
        
        poly_img = np.where(MA,poly_img,(ipoly+1))  

    del MA
    del T
    del _
    del overlap_polys
    
    gc.collect()
    
    
    
    
if were_there_overlaps:
    
    print("OVERLAPS HAVE BEEN DETECTED. SEE ABOVE FOR FULL REPORT")
    
    
    
    
    
    
    
    
    
    
    
            


print('')
print('Done !')


