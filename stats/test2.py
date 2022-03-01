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


input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
'/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
]


# all_pred_folder = '/mnt/c/Users/Gabriel/GeoData/Predictions/'
all_pred_folder = '/mnt/c/Users/Gabriel/Programmation/PyTorchLakeSegmentation/Segmentation_lakes_2/all_predictions/'



lake_area_sum_filename = "./sum_area.txt"

Recompute_area = True
# Recompute_area = False

# Fused_polys = False
Fused_polys = True


if Recompute_area:
    data_dict = {}
        
        
    for file_path in input_img_list:
        
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        base_img_folder = all_pred_folder+file_root+'/'
        
        if Fused_polys:
            
            base_poly_mul_folder = base_img_folder+'/polygons_mul_fused/'

            for file_path in os.listdir(base_poly_mul_folder):
                file_path = os.path.join(base_poly_mul_folder, file_path)
                the_file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                
                if (file_ext == '.shp' ):
                    
                    print(file_path)
                    
                    lake_outlines = gpd.read_file(file_path)

                    
                    training_name = the_file_root.split('/')[-1]
                    
                    data_dict[training_name] = sum(lake_outlines['Shape_Area'])

            
        else:
            

            base_poly_mul_folder = base_img_folder+'/polygons_mul/'


            training_folders = [os.path.join(base_poly_mul_folder, o) for o in os.listdir(base_poly_mul_folder) 
                                if os.path.isdir(os.path.join(base_poly_mul_folder,o))]

                
            print('\nReading Predictions\n')
            
            for the_training_folder in training_folders:
            
                init_lake_bou = the_training_folder+'/fused_polygons/all_polys.shp'
                print(init_lake_bou)
                
                
                lake_outlines = gpd.read_file(init_lake_bou)

                
                training_name = the_training_folder.split('/')[-1]
                
                data_dict[training_name] = sum(lake_outlines['Shape_Area'])


    json.dump(data_dict,open(lake_area_sum_filename,"w"))
    
else:
    data_dict = json.load(open(lake_area_sum_filename))
    
    
    
lists = sorted(data_dict.items())
x,y = zip(*lists)

plt.figure(figsize=(12,6))
ax = plt.gca()
plt.plot(x,y)

xmin, xmax = ax.get_xlim()
ax.hlines(y=146154000,xmin=xmin,xmax=xmax)


plt.xticks(rotation=90)
plt.ylabel("Combined lake area")
plt.tight_layout()
plt.savefig("fig.png")
