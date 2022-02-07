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
import pandas as pd
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




input_poly_folder = './polygons/polygons/'
output_poly_folder = './polygons/fused_polygons/'

# ~ one_image_filename = './input/all_images/Spot7_Syrdakh_BW0.TIF'
# ~ one_image_filename = '/mnt/c/Users/Gabriel/GeoData/Spot7_Syrdakh_BW.tif'
one_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/test1/DZB1216_19808.TIF'



for store_folder in [output_poly_folder]:
    if not(os.path.isdir(store_folder)):
        os.mkdir(store_folder)


image_filename_list = []
poly_filename_list = []

npoly_tot = 0

with rio.open(one_image_filename) as img_open:

    Big_GDF = gpd.GeoDataFrame(
                crs=img_open.crs
            )

    for file_path in os.listdir(input_poly_folder):
        file_path = os.path.join(input_poly_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.shp' ):
            
            print(file_path)
            
            lake_outlines = gpd.read_file(file_path)
            
            lake_outlines.to_crs(img_open.crs,inplace=True) 
            
            npoly = lake_outlines['geometry'].shape[0]
            
            for ipoly in range(npoly):
                lake_outlines.loc[ipoly,'Id'] = int(float(lake_outlines.loc[ipoly,'Id'])) + npoly_tot
                
            npoly_tot += npoly

            Big_GDF = gpd.GeoDataFrame( pd.concat( [Big_GDF,lake_outlines], ignore_index=True) ,crs=img_open.crs)
       
           
    output_poly_filename = output_poly_folder + 'all_polys_not_fused.shp'
    Big_GDF.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)

           
           
    # ~ print(Big_GDF['geometry'][0])
    Big_GDF.to_crs({'proj':'cea'},inplace=True) 
    # ~ print(Big_GDF['geometry'][0])

    # ~ Big_GDF.to_crs(img_open.crs,inplace=True) 
                    
    print(npoly_tot)

    # ~ buffer_distance = 12
    buffer_distance = 0

    print('Computing buffers')

    exp_geometry = []

    for ipoly in range(npoly_tot):
        print('buffer',ipoly,npoly_tot)
        
        if (buffer_distance == 0):
            exp_geometry.append(Big_GDF['geometry'][ipoly])
        else:
            exp_geometry.append(Big_GDF['geometry'][ipoly].buffer(buffer_distance))
        
    inter_ij = np.full((npoly_tot,npoly_tot),False,dtype=bool)

    print('Computing intersections')

    for ipoly in range(npoly_tot):
        print('intersection',ipoly,npoly_tot)
        inter_ij[ipoly,ipoly] = True
        for jpoly in range(ipoly+1,npoly_tot):
            inter_ij[ipoly,jpoly] = exp_geometry[ipoly].intersects(exp_geometry[jpoly])
            inter_ij[jpoly,ipoly] = inter_ij[ipoly,jpoly]
    
    print('Building Classes')
    
    eq_classes = []
    for ipoly in range(npoly_tot):
        print('class',ipoly,npoly_tot)
        direct_intersect = []
        for jpoly in range(npoly_tot):
            if inter_ij[ipoly,jpoly]:
                direct_intersect.append(jpoly)
        
        partial_class = set(direct_intersect)

        for the_class in eq_classes:
            
            if partial_class.intersection(the_class):
                
                partial_class.update(the_class)
        
        eq_classes = [ the_class for the_class in eq_classes if not(partial_class.intersection(the_class))]
        eq_classes.append(partial_class)
    
    print('Number of classes ',len(eq_classes))
    
    the_ids = []
    the_geometry = []

    print('Fusing polygons')

    for iclass in range(len(eq_classes)):
        print('fuse',iclass,len(eq_classes))
        
        class_polys = [Big_GDF['geometry'][ipoly] for ipoly in eq_classes[iclass]]
        
        for ipoly in eq_classes[iclass]:
        
            for jpoly in eq_classes[iclass]:
                
                if (ipoly != jpoly):
                    class_polys.append(exp_geometry[ipoly].intersection(exp_geometry[jpoly]))
        
        the_ids.append(iclass)
        the_geometry.append(shapely.ops.unary_union(class_polys))


    

    gdf_fused = gpd.GeoDataFrame(
        {'Id': the_ids, 'geometry': the_geometry },
        crs={'proj':'cea'}
    )
        
    gdf_fused['Shape_Area'] = gdf_fused.area
    gdf_fused.to_crs(img_open.crs,inplace=True) 

    output_poly_filename = output_poly_folder + 'all_polys.shp'
    gdf_fused.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)



print('')
print('Done !')

