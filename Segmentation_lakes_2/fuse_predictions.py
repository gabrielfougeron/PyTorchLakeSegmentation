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




# List of small images => nx_out = 1024
# input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_23_Spot4/SCENE01/2010_09_23_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4/130-01_4_286220_10-09-26-0238221M0/SCENE01/2010-09-26_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4c/SCENE01/2010_09_26_Spot4c.TIF',
# ]

# List of big images => nx_out = 2048
input_img_list = [
# '/mnt/c/GeoData/Polygon_Annotations/1980_DZB1216/1980_DZB1216.tif',
# '/mnt/c/GeoData/Polygon_Annotations/1989_07_12_Spot1/SCENE01/1989_07_12_Spot1.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/1989_07_16_Spot1/SCENE01/1989_07_16_Spot1.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/2005_09_25_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2007_08_02_Spot5/SCENE01/2007_08_02_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2011_09_11_Spot5/SCENE01/2011_09_11_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
'/mnt/c/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2013_08_23_Spot5/SCENE01/2013_08_23_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2013_08_23_Spot5b/SCENE01/2013_08_23_Spot5b.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2013_08_24_Spot5/SCENE01/2013_08_24_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2013_08_24_Spot5b/SCENE01/2013_08_24_Spot5b.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2016_Spot7/Spot7_Syrdakh_BW.tif',
# '/mnt/c/GeoData/Polygon_Annotations/extra/1989_07_16_Spot1b/SCENE01/1989_07_16_Spot1b.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/extra/2011_07_30_Spot4/SCENE01/2011_07_30_Spot4.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/extra/2013_08_24_Spot5c/SCENE01/2013_08_24_Spot5c.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/extra/2013_08_30_Spot5/SCENE01/2013_08_30_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/extra/2013_09_02_Spot5/SCENE01/2013_09_02_Spot5.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
]

for file_path in input_img_list:
    if not(os.path.isfile(file_path)):
        raise ValueError(f'This is not a valid file : {file_path}')



# target_mean = 127
target_mean = 120
target_stddev = 40
# target_stddev = 20


# nx_out = 4096
# ny_out = 4096

nx_out = 2048
ny_out = 2048

# nx_out = 1024
# ny_out = 1024

# nx_out = 512
# ny_out = 512


    
# nx_in = nx_out
# ny_in = ny_out

    
nx_in = 1024
ny_in = 1024


# Keep_Img_plots = True
Keep_Img_plots = False

# mod_img_lara_fix = True
mod_img_lara_fix = False

n_input_img = len(input_img_list)



all_pred_folder = './all_predictions/'

transfo_folder = './transformed_make_scenes/'
transfo_masks_folder = transfo_folder+'Lakes_masks/'
transfo_imgs_folder = transfo_folder+'Lakes_png_images/'


mask = np.zeros((nx_out,ny_out),dtype=np.uint8)
mask[0:(nx_out//2),0:(ny_out//2)]=1
PIL_mask = Image.fromarray(mask)
PIL_mask = PIL_mask.resize(size=(nx_in,ny_in),resample=Image.NEAREST)
PIL_mask.save(transfo_masks_folder+"/tmp_mask.png")


i_img = 0

for file_path in input_img_list:
    
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))

    i_img = i_img + 1
    print('')
    print('----------------------------------------------')
    print('')   
    print('Processing image '+str(i_img)+" of "+str(n_input_img))
    print(file_root)
    

    # Open raster image
    with rio.open(file_path) as img_open:
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

        del img

        print("image loaded")

        base_img_folder = all_pred_folder+file_root+'/'
        base_poly_mul_folder = base_img_folder+'/polygons_mul/'
        output_aggreg_poly_folder = base_img_folder+'/polygons_mul_fused/'

        for store_folder in [base_img_folder,base_poly_mul_folder,output_aggreg_poly_folder]:
            if not(os.path.isdir(store_folder)):
                os.makedirs(store_folder)
        


        # All model predictions done.
        # Now aggregate different predictions

        training_folders = [os.path.join(base_poly_mul_folder, o) for o in os.listdir(base_poly_mul_folder) 
                            if os.path.isdir(os.path.join(base_poly_mul_folder,o))]

        # print(training_folders)

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
            
            output_poly_filename = output_aggreg_poly_folder+'all_polys_'+str(n_pred).zfill(2)+'.shp'
                        
            gdf.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
            

        

            


print('')
print('Done !')


