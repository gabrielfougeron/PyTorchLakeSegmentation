# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:34:54 2021

@author: Lara
"""
import os
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

import torch





input_image_folder = './input/images/'
input_poly_folder = './input/polygons/'

output_masks_folder = './output/Lakes_masks'
output_imgs_folder = './output/Lakes_png_images'

ncutx = 4
ncuty = 4


image_filename_list = []
poly_filename_list = []


for file_path in os.listdir(input_image_folder):
    file_path = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))
    

    if (file_ext == '.TIF' ):
        
        poly_filename = input_poly_folder+file_root+'.shp'
        
        if os.path.exists(poly_filename):
            poly_filename_list.append(poly_filename)
            image_filename_list.append(input_image_folder+file_root+'.TIF')
        else:
            raise RuntimeError("Matching polygon file not found")

for store_folder in [output_masks_folder,output_imgs_folder]:
    if not(os.path.isdir(store_folder)):
        os.mkdir(store_folder)

n_img = len(image_filename_list)
        
for i_img in range(n_img):
    
    print('')
    print('----------------------------------------------')
    print('')
    print('Processing image '+str(i_img+1)+' of '+str(n_img))

    image_name = image_filename_list[i_img]
    lake_bou_name = poly_filename_list[i_img]
    
    print(image_name)
    print(lake_bou_name)

    # Open raster image
    with rio.open(image_name) as img_open:
        img = img_open.read()


    nx = img.shape[1]
    ny = img.shape[2]
    print('nx = ',nx)
    print('ny = ',ny)


    # Bounding box
    BB = plotting_extent(img_open)
    xmin,xmax,ymin,ymax = plotting_extent(img_open)
    print('xmin = ',xmin)
    print('xmax = ',xmax)
    print('ymin = ',ymin)
    print('ymax = ',ymax)


    #import lake boundaries
    lake_outlines = gpd.read_file(lake_bou_name)

    # Project lake boundaries to match Spot data
    lake_outline_match=lake_outlines.to_crs(img_open.crs)

    npoly = lake_outline_match['geometry'].shape[0]
    print('npoly = ',npoly)

    if (npoly > 255):
        warnings.warn("Too many polygons in a single image", RuntimeWarning)

    # ~ lake_outlines=lake_outlines.to_crs(img_open.crs)

    fig = plt.figure()
    ax=plt.gca()
    
    ep.plot_bands(img,ax=ax, cbar=False,extent=BB)
    lake_outlines.plot(ax=ax) 
    
    plt.savefig("out"+str(i_img)+".png")



print('')
print('Done !')









'''



















image_name = './input/images/Spot7_Syrdakh_BW3.tif'
lake_bou_name = './input/polygons/Spot7_Syrdakh_BW3.shp'


#import lake boundaries
lake_outlines = gpd.read_file(lake_bou_name)

# ~ print('This is the crs of lake_outlines:',(lake_outlines.crs))
print(type(lake_outlines))
# ~ print(lake_outlines)
# ~ print(lake_outlines['geometry'])
# ~ print(lake_outlines['geometry'].shape)

# ~ print(1/0)


# Open Spot data in read mode
with rio.open(image_name) as img_open:
    img = img_open.read()




print(type(img))
print(img.shape)
print(img.dtype)
print(img[0,0,0])



# Project lake boundaries to match Spot data
lake_outline_match=lake_outlines.to_crs(img_open.crs)


# Bounding box
BB = plotting_extent(img_open)

print(BB)


#Plot uncropped array
# ~ f,ax = plt.subplots()
fig = plt.figure()
ax=plt.gca()


# ~ ep.plot_rgb(Spot2016_data,
            # ~ rgb=(0,1,2),
            # ~ ax=ax,
            # ~ title ="Lake outlines overlaying Spot 2016",
            # ~ extent=Spot2016_plot_extent)

ep.plot_bands(img,ax=ax, cbar=False,extent=BB)
lake_outlines.plot(ax=ax) 
# ~ lake_outline_match.plot(ax=ax) 

 # ~ extent=



newpoint_np = np.array([[592686.0,   6917002.0]])

# ~ 590586.0, 593658.0, 6915702.0, 6918774.0)

# Turn points into list of x,y shapely points 
newpoint_list = [shapely.geometry.Point(xy) for xy in newpoint_np]



for pt in newpoint_list:
    
    for poly in lake_outlines['geometry']:
        
        # ~ print(type(poly))
        
       print(pt.within(poly))



newpoint_gpd = gpd.GeoDataFrame(newpoint_list, 
                                  columns=['geometry'],
                                  crs=img_open.crs)


newpoint_gpd.plot(ax=ax, 
                    color='springgreen', 
                    marker='*',
                    markersize=45)






# ~ plt.show()
plt.savefig("out.png")

# get plotting extent of raster data file





# ~ torch







'''





