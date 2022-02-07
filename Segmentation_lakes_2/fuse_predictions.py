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



# init_image = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/2005_09_25_Spot5.TIF'
# init_image = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF'
init_image = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF'


base_folder = './polygons_mul_stash/'

training_folders = [os.path.join(base_folder, o) for o in os.listdir(base_folder) 
                    if os.path.isdir(os.path.join(base_folder,o))]

output_folder = './polygons_mul_fused/'


with rio.open(init_image) as img_open:

    print('Reading Image\n')

    img = img_open.read()
    # img_uint8 = ((img[0,:,:])).astype(np.uint8)

    nx_img = img.shape[1]
    ny_img = img.shape[2]
    print('nx = ',nx_img)
    print('ny = ',ny_img)

    del img

    poly_img = np.zeros((nx_img,ny_img),dtype=np.uint8)
        
    print('\nReading Predictions\n')
    
    for the_training_folder in training_folders:
        
        init_lake_bou = the_training_folder+'/fused_polygons/all_polys.shp'
            

        lake_outlines = gpd.read_file(init_lake_bou)
        lake_outline_match=lake_outlines.to_crs(img_open.crs)

        npoly = lake_outline_match['geometry'].shape[0]
        print('npoly = ',npoly)

        MA,T,_ = rio.mask.raster_geometry_mask(img_open, lake_outline_match['geometry'], all_touched=False, invert=True)
        
        poly_img += np.where(MA,np.uint8(1),np.uint8(0))  
        
    
    n_pred_tot = len(training_folders)
    
    print('\nAggregating Predictions\n')
    
    for i in range(n_pred_tot):
        
        n_pred = np.uint8(n_pred_tot - i)

        poly_img_inter = np.where(poly_img >= n_pred,np.uint8(1),np.uint8(0))
        
        
        polyval = []
        geometry = []

        the_win = rio.windows.Window(0,0,nx_img,ny_img)
        the_transform = img_open.window_transform(the_win)
        the_polys = rio.features.shapes(poly_img_inter, transform=img_open.transform)
        
        npoly = 0
        
        for shapedict, value in the_polys:

            if (value != 0):
                
                polyval.append(str(npoly))
                geometry.append(shapely.geometry.shape(shapedict))
                
                npoly += 1 
        
        
        print('npoly = ',npoly)
        
        # build the gdf object over the two lists
        gdf = gpd.GeoDataFrame(
            {'Id': polyval, 'geometry': geometry },
            crs=img_open.crs
        )
        
        gdf.to_crs({'proj':'cea'},inplace=True) 
        
        gdf['Shape_Area'] = gdf.area
        
        gdf.to_crs(img_open.crs,inplace=True) 
        
        output_poly_filename = output_folder+'all_polys_'+str(n_pred).zfill(2)+'.shp'
                    
        gdf.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
        

    
    
            


print('')
print('Done !')


