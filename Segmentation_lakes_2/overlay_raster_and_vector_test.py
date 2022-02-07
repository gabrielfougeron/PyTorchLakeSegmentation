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
# ~ import earthpy as et
# ~ import earthpy.spatial as es
import earthpy.plot as ep
import fiona

import shapely

import torch




input_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/UniversiteParisSud_A.SEJOURNE_SO20004818-44-01_52862211010030255002B0/SCENE01/IMAGERY.TIF'
# ~ input_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/UniversiteParisSud_A.SEJOURNE_SO20004818-131-01_42862211009260238301M0/SCENE01/IMAGERY.TIF'
# ~ input_image_filename = './polygons/test/Spot7_Syrdakh_BW105.TIF'
# ~ input_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/test1/DZB1216_19808.TIF'
# ~ input_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/test1/DZB1216_198010.TIF'
# ~ input_image_filename = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/test2/Spot7_Syrdakh_BW105.TIF'
input_poly_filename = './polygons/fused_polygons/all_polys.shp'

# Open raster image
with rio.open(input_image_filename) as img_open:
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


    # ~ print(img_open.bounds)


    #import lake boundaries
    lake_outlines = gpd.read_file(input_poly_filename)

    # Project lake boundaries to match Spot data
    lake_outline_match=lake_outlines.to_crs(img_open.crs)
    # ~ lake_outline_match=lake_outlines

    npoly = lake_outline_match['geometry'].shape[0]
    print('npoly = ',npoly)

    fig = plt.figure(figsize=(10,10))
    ax=plt.gca()

    # ~ ep.plot_bands(img,ax=ax, cbar=False,extent=BB)
    rio.plot.show(img, transform=img_open.transform,ax=ax,cmap='gray')

    lake_outlines.plot(ax=ax) 

    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)


    plt.savefig("out.png")
    
    
    # ~ for poly in lake_outline_match['geometry']:
        
        # ~ x, y = poly.exterior.coords.xy
        # ~ print(min(y),max(y))
        # ~ print(y)
                                



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





