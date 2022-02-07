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
import geopandas as gpd
import rasterio as rio
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import fiona

import shapely
import rasterstats


import torch
import warnings

from PIL import Image

def inBB(BB,x,y):
    return (x > BB[0]) and (y > BB[1]) and (x < BB[2]) and (y < BB[3])


input_image_folder = './input/images/'
input_poly_folder = './input/polygons/'

output_masks_folder = './output/Lakes_masks'
output_imgs_folder = './output/Lakes_png_images'

output_render_folder = './render'

# ~ ncutx = 4
# ~ ncuty = 4

ncutx = 1
ncuty = 1


image_filename_list = []
poly_filename_list = []


for file_path in os.listdir(input_poly_folder):
    file_path = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))
    
    if (file_ext == '.shp' ):
        
        image_filename = input_image_folder+file_root+'.TIF'
        
        if os.path.exists(image_filename):
            
            image_filename_list.append(image_filename)
            
            poly_filename = input_poly_folder+file_root+'.shp'
            poly_filename_list.append(poly_filename)
            
        else:
            print('')
            print(image_filename)
            raise RuntimeError("Matching image file not found")

for store_folder in [output_masks_folder,output_imgs_folder,output_render_folder]:
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
        
        # ~ if ('Shape_Area' in  lake_outline_match.keys()) :
            # ~ print(lake_outline_match['Shape_Area'])
            # ~ print(lake_outline_match.area)
        

        npoly = lake_outline_match['geometry'].shape[0]
        print('npoly = ',npoly)

        if (npoly > 255):
            warnings.warn("Too many polygons in a single image", RuntimeWarning)

        polyBB = np.zeros((npoly,4))
        for ipoly in range(npoly):
            polyBB[ipoly,:] = lake_outline_match['geometry'][ipoly].bounds
            

        # ~ print(img.dtype)
        # ~ print(np.amax(img))
        # ~ print(np.amin(img))
        # ~ mask = (((img[0,:,:] - np.amin(img))/(np.amax(img) - np.amin(img))) * 255).astype(np.uint8)

        # ~ img_uint8 = (img[0,:,:]/16).astype(np.uint8)
        # ~ img_uint8 = (np.minimulmmin(img[0,:,:]/4,255)).astype(np.uint8)
        img_uint8 = ((img[0,:,:]/4)).astype(np.uint8)



        mask = np.zeros((nx,ny),dtype=np.uint8)
                
        for ipoly in range(npoly):

            # ~ MA,T,_ = rio.mask.raster_geometry_mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)
            MA,T = rio.mask.mask(img_open, [lake_outline_match['geometry'][ipoly]], all_touched=False, invert=True)

            mask = mask + np.where(MA,ipoly,0).astype(np.uint8)
            # ~ mask = mask + np.where(MA,255,0).astype(np.uint8)


        dx = nx // ncutx
        dy = ny // ncuty

        for ix in range(ncutx):
            for iy in range(ncuty):
                
                    
                cut_mask = mask[ix*dx:(ix+1)*dx,iy*dy:(iy+1)*dy]
                obj_ids, obj_counts = np.unique(cut_mask, return_counts=True)
                
                pxl_thresh = 4
                n_obj = obj_ids.shape[0]
                
                n_obj_thr = 0
                
                for i in range(n_obj):
                    if (obj_counts[i] >= pxl_thresh):
                        n_obj_thr += 1
                
                obj_ids_thr = np.zeros((n_obj_thr),dtype=obj_ids.dtype)
                obj_counts_thr = np.zeros((n_obj_thr),dtype=obj_counts.dtype)
                
                n_obj_thr = 0
                
                for i in range(n_obj):
                    if (obj_counts[i] >= pxl_thresh):
                
                        obj_ids_thr[n_obj_thr] = obj_ids[i]
                        obj_counts_thr[n_obj_thr] = obj_counts[i]

                        n_obj_thr += 1

                # ~ print(i_img,ix,iy)
                # ~ print(obj_counts_thr)
                # ~ print('')
                
                if (obj_ids_thr.size > 1):
                    

                    # ~ PIL_img = Image.fromarray(img_uint8[ix*dx:(ix+1)*dx,iy*dy:(iy+1)*dy])
                    # ~ PIL_img.save(output_imgs_folder+"/img_"+str(i_img)+'_'+str(ix)+"_"+str(iy)+".png")
                        
                    # ~ PIL_mask = Image.fromarray(mask[ix*dx:(ix+1)*dx,iy*dy:(iy+1)*dy])
                    # ~ PIL_mask.save(output_masks_folder+"/mask_"+str(i_img)+'_'+str(ix)+"_"+str(iy)+".png")
                    
                        
                    PIL_img = Image.fromarray(img_uint8[ix*dx:(ix+1)*dx,iy*dy:(iy+1)*dy])
                    PIL_img.save(output_imgs_folder+"/img_"+str(i_img)+'_'+str(ix)+"_"+str(iy)+".png")
                    
                    obj_ids_thr = np.sort(obj_ids_thr)
                    new_mask = np.searchsorted(obj_ids_thr,cut_mask).astype(np.uint8)
                        
                    PIL_mask = Image.fromarray(new_mask)
                    PIL_mask.save(output_masks_folder+"/mask_"+str(i_img)+'_'+str(ix)+"_"+str(iy)+".png")






    fig = plt.figure(figsize=(10,10))
    ax=plt.gca()

    ep.plot_bands(img,ax=ax, cbar=False,extent=BB)
    # ~ lake_outlines.plot(ax=ax) 
    lake_outline_match.plot(ax=ax) 

    plt.tight_layout()
    # ~ plt.show()
    plt.savefig(output_render_folder+"/out"+str(i_img)+".png")



print('')
print('Done !')
