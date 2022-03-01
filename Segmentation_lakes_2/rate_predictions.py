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

import pandas as pd

import shapely
import rasterstats

import gc

import torch
import warnings

from scipy import ndimage

from PIL import Image


def bar_plot(ax, labels,data, colors=None, total_width=0.8, single_width=1):
    
    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    for i in range(n_bars):
        this_data = data[i]
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        
        for j in range(len(this_data)):
            bar = ax.bar(j + x_offset, this_data[j], width=bar_width * single_width, color=colors[i % len(colors)])
            
    plt.xticks(ticks = [i for i in range(len(labels))],labels = labels)
        



input_img_list = [
'/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
]


reference_shp_list = [
# '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2012_09_25_NEW/2012_09_25_lakes.shp',
'/mnt/c/Users/Gabriel/GeoData/Exact_Lakes/2012-09-25/2012_09_25_lakes.shp',
# '/mnt/c/Users/Gabriel/GeoData/Exact_Lakes/2013-07-14/2013_07_14_lakes.shp',
]







for file_path in input_img_list:
    if not(os.path.isfile(file_path)):
        raise ValueError(f'This is not a valid file : {file_path}')


nx_out = 2048
ny_out = 2048
    
nx_in = 1024
ny_in = 1024


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



Aggregated_predictions = True
# Aggregated_predictions = False 

# Initial_predictions = True
Initial_predictions = False

for i_img in range(len(input_img_list)):
    
    file_path = input_img_list[i_img]
    
    img_file_root, file_ext = os.path.splitext(os.path.basename(file_path))

    print('')
    print('----------------------------------------------')
    print('')   
    print('Processing image '+str(i_img+1)+" of "+str(n_input_img))
    print(img_file_root)
    

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
        
        
        print("Loading reference shapefile")
  
        lake_outlines_ref = gpd.read_file(reference_shp_list[i_img])
        lake_outlines_ref_match=lake_outlines_ref.to_crs(img_open.crs)

        npoly = lake_outlines_ref_match['geometry'].shape[0]
        print('npoly = ',npoly)

        MA_REF,T,_ = rio.mask.raster_geometry_mask(img_open, lake_outlines_ref_match['geometry'], all_touched=False, invert=True)
        
        pxl_ref = np.sum(np.where(MA_REF,np.uint16(1),np.uint16(0)))

        print(f'pxl_ref = {pxl_ref}')



        pred_files = []
        nice_name = []
    
            
        base_img_folder = all_pred_folder+img_file_root+'/'
        base_poly_mul_folder = base_img_folder+'/polygons_mul/'
        output_aggreg_poly_folder = base_img_folder+'/polygons_mul_fused/'

        
        if (Aggregated_predictions):
    
            for file_path in os.listdir(output_aggreg_poly_folder):
                file_path = os.path.join(output_aggreg_poly_folder, file_path)
                the_file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                
                if (file_ext == '.shp' ):
                    
                    pred_files.append(file_path)
                    nice_name.append(the_file_root)

        if (Initial_predictions):

            training_folders = [os.path.join(base_poly_mul_folder, o) for o in os.listdir(base_poly_mul_folder) 
                                if os.path.isdir(os.path.join(base_poly_mul_folder,o))]

            for the_training_folder in training_folders:
                
                init_lake_bou = the_training_folder+'/fused_polygons/all_polys.shp'
                
                pred_files.append(init_lake_bou)
                nice_name.append(init_lake_bou.split('/')[-3])
              
        all_false_pos_rate = []
        all_false_neg_rate = []
        all_disagree_rate = []
                
        for lake_bou in pred_files:
    
            print('')
            print(lake_bou)
                
            lake_outlines = gpd.read_file(lake_bou)
            lake_outline_match=lake_outlines.to_crs(img_open.crs)

            npoly = lake_outline_match['geometry'].shape[0]
            # print('npoly = ',npoly)

            MA,T,_ = rio.mask.raster_geometry_mask(img_open, lake_outline_match['geometry'], all_touched=False, invert=True)
            
            pxl_false_pos = np.sum(np.where(np.logical_and(MA    ,np.logical_not(MA_REF)),np.uint16(1),np.uint16(0)))
            pxl_false_neg = np.sum(np.where(np.logical_and(MA_REF,np.logical_not(MA    )),np.uint16(1),np.uint16(0)))
            pxl_disagree = pxl_false_pos + pxl_false_neg
            
            false_pos_rate = float(pxl_false_pos)/float(pxl_ref)
            false_neg_rate = float(pxl_false_neg)/float(pxl_ref)
            disagree_rate = float(pxl_false_pos+pxl_false_neg)/float(pxl_ref)

            all_false_pos_rate.append(false_pos_rate)
            all_false_neg_rate.append(false_neg_rate)
            all_disagree_rate.append(disagree_rate)

            print(f'false_pos_rate = {false_pos_rate}')
            print(f'false_neg_rate = {false_neg_rate}')
            print(f'disagree_rate = {disagree_rate}')
        
        excel_filename_out = os.path.join(base_img_folder,'out.xlsx')
        
        the_data = list(map(list, zip(nice_name, all_false_pos_rate, all_false_neg_rate, all_disagree_rate)))
        
        df = pd.DataFrame(
        the_data,
        columns =['Name', 'False positive rate', 'False negative rate','Disagree rate']
        # , dtype = float
        )
        
        df.to_excel(excel_filename_out)  
        
        fig = plt.figure(figsize=(16,8))
        ax = fig.gca()
        
        bar_plot(ax, nice_name,[all_false_pos_rate,all_false_neg_rate,all_disagree_rate] )
        
        plt.xticks(rotation=90)
        output_filename = os.path.join(base_img_folder,'Relative_error.png')

        plt.savefig(output_filename,bbox_inches='tight', pad_inches=0)
        plt.close()


print('')
print('Done !')


