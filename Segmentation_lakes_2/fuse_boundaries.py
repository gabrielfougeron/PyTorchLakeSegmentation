# -*- coding: utf-8 -*
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

import shapely
import rasterstats

import shutil
import warnings


all_bou_folder = '/mnt/c/GeoData/All_images_boundaries'

output_aggreg_poly_folder = './fused_boundaries'

all_shapefiles = []

fake_tif_name = 'fake.tif'
create_fake_tif = True
# create_fake_tif = False

ref_crs = 'epsg:32652'

nx_img = 1024
ny_img = 1024
poly_img = np.zeros((nx_img,ny_img),dtype=np.uint8)
    
print('Listing shapefiles')
for the_file in os.listdir(all_bou_folder):

    base,ext =  os.path.splitext(the_file)

    if ext == '.shp' : 

        all_shapefiles.append(os.path.join(all_bou_folder,the_file))


xmin =  1e100
xmax = -1e100
ymin =  1e100
ymax = -1e100

print('Computing AABB')
for shapefile in all_shapefiles :
# 
#     print('')
#     print(shapefile)

    image_bou = gpd.read_file(shapefile)
    image_bou = image_bou.to_crs(ref_crs)

    aabbs = image_bou.bounds

    for index, row in aabbs.iterrows():

        xmin = min(xmin,row['minx'])
        xmax = max(xmax,row['maxx'])
        ymin = min(ymin,row['miny'])
        ymax = max(ymax,row['maxy'])

main_transform = rio.transform.from_bounds(xmin, ymin, xmax, ymax, nx_img, ny_img)
# main_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax)
   
# Create fake tif if necessary

if create_fake_tif:
    new_dataset = rio.open(fake_tif_name, 'w', driver='GTiff',
                                height = poly_img.shape[0], width = poly_img.shape[1],
                                count=1, dtype=str(poly_img.dtype),
                                crs=ref_crs,
                                transform=main_transform)
    new_dataset.write(poly_img, 1)
    new_dataset.close()


# Open fake tif
with rio.open(fake_tif_name) as img_open:

    print('Reading Predictions')
    for shapefile in all_shapefiles :

        image_bou = gpd.read_file(shapefile)
        image_bou = image_bou.to_crs(ref_crs)

        MA,T,_ = rio.mask.raster_geometry_mask(img_open, image_bou['geometry'], all_touched=False, invert=True)
        
        poly_img += np.where(MA,np.uint8(1),np.uint8(0))  
        
    n_pred_tot = len(all_shapefiles)
    
    print('Aggregating Predictions')

    for i in range(n_pred_tot):
        
        n_pred = np.uint8(n_pred_tot - i)

        poly_img_inter = np.where(poly_img >= n_pred,np.uint8(1),np.uint8(0))
                
        polyval = []
        geometry = []

        the_polys = rio.features.shapes(poly_img_inter, transform=img_open.transform)
        
        npoly = 0
        
        for shapedict, value in the_polys:

            if (value != 0):
                
                polyval.append(str(npoly))
                geometry.append(shapely.geometry.shape(shapedict))
                
                npoly += 1 
        
        
        print(i,'npoly = ',npoly)
        
        if (npoly > 0):

            # build the gdf object over the two lists
            gdf = gpd.GeoDataFrame(
                {'Id': polyval, 'geometry': geometry },
                crs=img_open.crs
            )
            
            gdf.to_crs({'proj':'cea'},inplace=True) 
            
            gdf['Shape_Area'] = gdf.area
            
            gdf.to_crs(img_open.crs,inplace=True) 
            
            output_poly_filename = os.path.join(output_aggreg_poly_folder,'all_polys_'+str(n_pred).zfill(2)+'.shp')
                        
            gdf.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
            


print('')
print('Done !')


