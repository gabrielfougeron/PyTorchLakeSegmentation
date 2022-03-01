# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:34:54 2021

@author: Lara
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio as rio
import json


# input_poly_folder = '/mnt/c/Users/Gabriel/GeoData/Predictions/2012_07_25_Spot5/polygons_mul_fused'
# input_poly_folder = '/mnt/c/Users/Gabriel/Programmation/PyTorchLakeSegmentation/Segmentation_lakes_2/all_predictions/2012_07_25_Spot5/polygons_mul_fused'
input_poly_folder = '/mnt/c/Users/Gabriel/Programmation/PyTorchLakeSegmentation/Segmentation_lakes_2/all_predictions/2012_09_25_Spot5/polygons_mul_fused'
# input_poly_folder = '/mnt/c/Users/Gabriel/GeoData/Predictions/2012_09_25_Spot5/polygons_mul_fused'
lake_area_sum_filename = "./sum_area.txt"

Recompute_area = True
# Recompute_area = False


if Recompute_area:
    data_dict = {}

    for file_path in os.listdir(input_poly_folder):
        file_path = os.path.join(input_poly_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        

        training_folders = [os.path.join(base_poly_mul_folder, o) for o in os.listdir(base_poly_mul_folder) 
                            if os.path.isdir(os.path.join(base_poly_mul_folder,o))]

        poly_img = np.zeros((nx_img,ny_img),dtype=np.uint8)
            
        print('\nReading Predictions\n')
        
        for the_training_folder in training_folders:
        
            init_lake_bou = the_training_folder+'/fused_polygons/all_polys.shp'
                
            # lake_outlines = gpd.read_file(init_lake_bou)
            print(init_lake_bou)
            
            # lake_outlines = gpd.read_file(file_path)
            
            # data_dict[file_root] = sum(lake_outlines['Shape_Area'])


    # json.dump(data_dict,open(lake_area_sum_filename,"w"))
    
# else:
    # data_dict = json.load(open(lake_area_sum_filename))
    
    
    
# lists = sorted(data_dict.items())
# x,y = zip(*lists)

# plt.figure(figsize=(12,6))
# ax = plt.gca()
# plt.plot(x,y)

# xmin, xmax = ax.get_xlim()
# ax.hlines(y=146154000,xmin=xmin,xmax=xmax)


# plt.xticks(rotation=90)
# plt.ylabel("Combined lake area")
# plt.tight_layout()
# plt.savefig("fig.png")
