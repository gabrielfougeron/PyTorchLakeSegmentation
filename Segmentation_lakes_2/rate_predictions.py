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
import matplotlib
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
        



### TRAINING DATA ###

# input_img_list = [
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2012_09_25/SCENE01/2012_09_25_Spot5.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2016_Spot7/Spot7_Syrdakh_BW.tif',
# ]

# reference_shp_list = [
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03_North_clipB.shp',
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/2010_10_03_lakes_clipped.shp',
# '/mnt/c/GeoData/Exact_Lakes/2012-09-25/2012_09_25_lakes.shp',
# '/mnt/c/GeoData/training_gab_01_23_2022/2016_Spot7/2016_Spot7.shp',
# ]



### VALIDATION DATA ###

input_img_list = [
'/mnt/c/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
]

reference_shp_list = [
'/mnt/c/GeoData/Test_ref_shapefile_2013_07_14_lakes/2013_07_14_lakes.shp',
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


Aggregated_predictions = True
# Aggregated_predictions = False 

# Initial_predictions = True
Initial_predictions = False


if Aggregated_predictions and Initial_predictions:
    output_files_name = 'all_predictions'
elif Aggregated_predictions:
    output_files_name = 'fused_predictions'
elif Initial_predictions:
    output_files_name = 'individual_predictions'
else:
    raise ValueError('Wrong output definition')




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
           
        print("Image shape : ",img.shape)
            
        nx_img = img.shape[1]
        ny_img = img.shape[2]
        print('nx = ',nx_img)
        print('ny = ',ny_img)
        total_image_pxl = nx_img*ny_img
        print(f'total_image_pxl : {total_image_pxl}')
                
        BB = plotting_extent(img_open)
        xmin,xmax,ymin,ymax = plotting_extent(img_open)
        # xmin,ymin,xmax,ymax = img_open.bounds
        print('xmin = ',xmin)
        print('xmax = ',xmax)
        print('ymin = ',ymin)
        print('ymax = ',ymax)
        
        total_image_area = (xmax-xmin)*(ymax-ymin)
        
        
        del img

        print("image loaded")
        
        
        print("Loading reference shapefile")
  
        lake_outlines_ref = gpd.read_file(reference_shp_list[i_img])
        lake_outlines_ref_match=lake_outlines_ref.to_crs(img_open.crs)

        npoly = lake_outlines_ref_match['geometry'].shape[0]
        print('npoly = ',npoly)

        MA_REF,T,_ = rio.mask.raster_geometry_mask(img_open, lake_outlines_ref_match['geometry'], all_touched=False, invert=True)
        
        pxl_ref = np.sum(np.where(MA_REF,1,0))

        print(f'pxl_ref = {pxl_ref}')
        print(f'pxl_ref proportion = {pxl_ref/total_image_pxl}')

        area_ref = sum(lake_outlines_ref_match['Shape_Area'])

        print(f'area_ref = {area_ref}')
        print(f'area_ref proportion = {area_ref/total_image_area}')


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
            
            pxl_pred = np.sum(np.where(MA,1,0))
            
            pxl_false_pos = np.sum(np.where(np.logical_and(MA    ,np.logical_not(MA_REF)),1,0))
            pxl_false_neg = np.sum(np.where(np.logical_and(MA_REF,np.logical_not(MA    )),1,0))
            pxl_disagree = pxl_false_pos + pxl_false_neg
            
            area = sum(lake_outline_match['Shape_Area'])
                
            false_pos_rate = float(pxl_false_pos)/float(pxl_ref)
            false_neg_rate = float(pxl_false_neg)/float(pxl_ref)
            disagree_rate = float(pxl_false_pos+pxl_false_neg)/float(pxl_ref)

            all_false_pos_rate.append(false_pos_rate)
            all_false_neg_rate.append(false_neg_rate)
            all_disagree_rate.append(disagree_rate)

            print(f'false_pos_rate = {false_pos_rate}')
            print(f'false_neg_rate = {false_neg_rate}')
            print(f'disagree_rate = {disagree_rate}')
            
            print(f'pxl_rate = {false_pos_rate - false_neg_rate}')
            print(f'pxl_rate = {(pxl_pred - pxl_ref)/pxl_ref}')

            print(f'area_rate = {(area - area_ref)/area_ref}')
            

            print(f'pxl_pred = {pxl_pred}')
            pxl_pred_prop = pxl_pred/total_image_pxl
            print(f'pxl_pred proportion = {pxl_pred_prop}')
            
            # print(f'area = {area}')
            
            area_prop = area/total_image_area
            print(f'area proportion     = {area_prop}')
            print(f'Relative error : {2*(pxl_pred_prop - area_prop) / (pxl_pred_prop + area_prop)}')
            
            
        
        excel_filename_out = os.path.join(base_img_folder,output_files_name+'.xlsx')
        
        the_data = list(map(list, zip(nice_name, all_false_pos_rate, all_false_neg_rate, all_disagree_rate)))
        
        df = pd.DataFrame(
        the_data,
        columns =['Name', 'False positive rate', 'False negative rate','Disagree rate']
        # , dtype = float
        )
        
        df.to_excel(excel_filename_out)  
        
        fig = plt.figure(figsize=(8,6))
        # fig = plt.figure(figsize=(16,12))
        ax = fig.gca()
        
        bar_plot(ax, nice_name,[all_false_pos_rate,all_false_neg_rate,all_disagree_rate] )
        
        ax.set_ylabel('Relative area wrt reference')
        # ax.set_ylim([0, 0.15])
        ax.set_ylim([0, 0.2])
        # ax.set_ylim([0, 0.5])
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                
        # custom_lines = [matplotlib.lines.Line2D([0], [0], color=colors[i], lw=10) for i in range(3)]
        # ax.legend(custom_lines, ['False Positive Rate', 'False Negative Rate', 'False Prediction Rate'])
        
        
        plt.xticks(rotation=90)
        
        plt.tight_layout(pad=2.)    
        
        output_filename = os.path.join(base_img_folder,output_files_name+'.png')
        plt.savefig(output_filename)
        output_filename = os.path.join(base_img_folder,output_files_name+'.pdf')
        plt.savefig(output_filename)
        
        plt.close()


print('')
print('Done !')


