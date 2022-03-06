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

import shapely
import rasterstats

import shutil


import warnings

import torch
from PIL import Image,ImageFilter

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import *
import utils
import transforms as T

from tv_training_code import *


device = torch.device('cuda')




# List of small images => nx_out = 1024
# input_img_list = [
# '/mnt/c/GeoData/Polygon_Annotations/2010_09_23_Spot4/SCENE01/2010_09_23_Spot4.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2010_09_26_Spot4/130-01_4_286220_10-09-26-0238221M0/SCENE01/2010-09-26_Spot4.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2010_09_26_Spot4c/SCENE01/2010_09_26_Spot4c.TIF',
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
# '/mnt/c/GeoData/Polygon_Annotations/extra/Archive/C1980_test2.tif',
# '/mnt/c/GeoData/Polygon_Annotations/2011_09_08_Spot5/56-01_5_289222_11-09-08-0213592B0/SCENE01/2011_09_08_Spot5.TIF',
]

# input_img_list = [
# '/mnt/c/GeoData/Polygon_Annotations/22.03.01/2010_10_03_Spot5b/SCENE01/2010_10_03_Spot5b.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/22.03.01/2013_07_19_Spot5/SCENE01/2013_07_19.TIF',
# ]

# input_img_list = [
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
# '/mnt/c/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
# ]


# input_img_list = [
# '/mnt/c/GeoData/Polygon_Annotations/1989_07_12_Spot1/SCENE01/1989_07_12_Spot1.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2007_08_02_Spot5/SCENE01/2007_08_02_Spot5.TIF',
# ]
    


for file_path in input_img_list:
    if not(os.path.isfile(file_path)):
        raise ValueError(f'This is not a valid file : {file_path}')



# target_mean = 127
# target_mean = 160
# target_mean = 120
target_mean = 120
# target_mean = 140
# target_stddev = 60
target_stddev = 40
# target_stddev = 60
# target_stddev = 80
# target_stddev = 120
# target_stddev = 140
# target_stddev = 20

# Local_contrast = False
Local_contrast = True


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


model_list = [
# "./trainings/4_RGB_ADAMS_RESTART/Restart_026.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_040.pt",
"./trainings/4_RGB_ADAMS_RESTART/Restart_049.pt",
]



# model_list = [
# "./trainings/4_RGB_ADAMS_RESTART/Restart_000.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_001.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_002.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_003.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_004.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_005.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_006.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_007.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_008.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_009.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_010.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_011.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_012.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_013.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_014.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_015.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_016.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_017.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_018.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_019.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_020.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_021.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_022.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_023.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_024.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_025.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_026.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_027.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_028.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_029.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_030.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_031.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_032.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_033.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_034.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_035.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_036.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_037.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_038.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_039.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_040.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_041.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_042.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_043.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_044.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_045.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_046.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_047.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_048.pt",
# "./trainings/4_RGB_ADAMS_RESTART/Restart_049.pt",
# ]



Keep_Img_plots = True
# Keep_Img_plots = False

mod_img_lara_fix = True
# mod_img_lara_fix = False

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
        
        if mod_img_lara_fix:
            # fix for cropped images
            print(img.dtype)
            img = np.where(img[0,:,:] == 256 ,np.uint16(0),img)
            
        if Local_contrast:
            
            img_uint8 = img[0,:,:].astype(np.uint8)
            
            del img
            
        # else:
                
            # vals, count = np.unique(img , return_counts=True)
            
            # vals = vals[1:]
            # count = count[1:]
            
            # print(vals)
            
            # mean = np.sum(vals*count)/np.sum(count)
            # stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
          
            # img_new = ((img.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
            
            # img_new = np.where(img_new > 255.,255.,img_new)
            # img_new = np.where(img_new < 0.,0.,img_new)
            # img_new = img_new.astype(np.uint8)

            
            # img_uint8 = np.where(img[0,:,:] == 0 ,np.uint8(0),img_new[0,:,:])

            # del img
            # del img_new


            
        else:
            
            nskew = 10
            
            vals, count = np.unique(img , return_counts=True)
            
            vals = vals[1:]
            count = count[1:]
            
            print(vals)
            
            mean = np.sum(vals*count)/np.sum(count)
            stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
          
            img_new = ((img.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
            
            img_new = np.where(img_new > 255-nskew,255-nskew,img_new)
            img_new = np.where(img_new < 0.,0.,img_new)
            img_new = img_new.astype(np.uint8)
            img_new = img_new + np.uint8(nskew)
            
            img_uint8 = np.where(img[0,:,:] == 0 ,np.uint8(0),img_new[0,:,:])

            del img
            del img_new




        print("image loaded")

        base_img_folder = all_pred_folder+file_root+'/'
        base_poly_mul_folder = base_img_folder+'/polygons_mul/'
        output_aggreg_poly_folder = base_img_folder+'/polygons_mul_fused/'

        for store_folder in [base_img_folder,base_poly_mul_folder,output_aggreg_poly_folder]:
            if not(os.path.isdir(store_folder)):
                os.makedirs(store_folder)
        
        filename_path_save = base_img_folder+'img_path.txt'
        
        # if os.path.isfile(filename_path_save):
            # raise ValueError(f'File {filename_path_save} already exists')
        
        with open(filename_path_save, 'w') as outfile:
            outfile.write(file_path)

        for the_model in model_list:
            
            the_basename = os.path.basename(the_model)
            model_root,ext = os.path.splitext(the_basename)
            
            print(model_root,ext)
            
            base_model_folder =base_poly_mul_folder+model_root+'/'

            output_poly_folder = base_model_folder +'polygons/'
            output_fused_poly_folder = base_model_folder+'fused_polygons/'
            output_imgs_folder = base_model_folder+'images/'

            for store_folder in [transfo_folder,transfo_masks_folder,transfo_imgs_folder,output_imgs_folder,output_poly_folder,output_fused_poly_folder]:
                if not(os.path.isdir(store_folder)):
                    os.makedirs(store_folder)

            model = torch.load(the_model)
            model.to(device)

            nxtot = img_uint8.shape[0]
            nytot = img_uint8.shape[1]

            xstart_list = [0,nx_out//2]
            ystart_list = [0,ny_out//2]

            # xstart_list = [0]
            # ystart_list = [0]
            
            istartmin = 0
            istartmax = len(xstart_list)
            
            ixmin = 0
            iymin = 0
            
            for istart in range(istartmin,istartmax):
                
                xstart = xstart_list[istart]
                ystart = ystart_list[istart]

                ixmax = (nxtot-xstart)//nx_out  + 1
                iymax = (nytot-ystart)//ny_out  + 1

                # ixmin = 3
                # iymin = 3
                # ixmax = 5
                # iymax = 5


                for ix in range(ixmin,ixmax):
                    for iy in range(iymin,iymax):
                        
                        print(model_root,' ',istart,istartmax,ix,ixmax,iy,iymax)

                        # img_uint8_small = img_uint8[xstart+ix*nx_out:xstart+(ix+1)*nx_out,ystart+iy*ny_out:ystart+(iy+1)*ny_out]
                        
                        xi = xstart+ix*nx_out
                        xf = xstart+(ix+1)*nx_out
                        yi = ystart+iy*ny_out
                        yf = ystart+(iy+1)*ny_out
                        img_uint8_small = np_safe_copy(img_uint8,xi,xf,yi,yf)
                        
                        
                        if Local_contrast:


                            nskew = 10

                           
                            vals, count = np.unique(img_uint8_small , return_counts=True)
                            
                            vals = vals[1:]
                            count = count[1:]
                            
                            mean = np.sum(vals*count)/np.sum(count)
                            stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))

                            img_uint8_small_new = ((img_uint8_small.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
                            
                            img_uint8_small_new = np.where(img_uint8_small_new > 255.-nskew,255.-nskew,img_uint8_small_new)
                            img_uint8_small_new = np.where(img_uint8_small_new < 0.,0.,img_uint8_small_new)
                            img_uint8_small_new = img_uint8_small_new.astype(np.uint8)
                            img_uint8_small_new = img_uint8_small_new + np.uint8(nskew)


                            img_uint8_small_new = np.where(img_uint8_small == 0 ,np.uint8(0),img_uint8_small_new)

                            
                            PIL_img = Image.fromarray(img_uint8_small_new)
                            PIL_img = PIL_img.resize(size=(nx_in,ny_in),resample=Image.BICUBIC)
                            
                        else:
                            
                            PIL_img = Image.fromarray(img_uint8_small)
                            PIL_img = PIL_img.resize(size=(nx_in,ny_in),resample=Image.BICUBIC)
                            
                        
                        # PIL_img = PIL_img.filter(ImageFilter.BLUR)
                        # PIL_img = PIL_img.filter(ImageFilter.SMOOTH_MORE)
                        # PIL_img = PIL_img.filter(ImageFilter.SMOOTH_MORE)
                        
                        
                        PIL_img.save(transfo_imgs_folder+"/tmp_img.png")
                        
                        # mask = np.zeros((nx_out,ny_out),dtype=np.uint8)
                        # mask[0:(nx_out//2),0:(ny_out//2)]=1
                        # PIL_mask = Image.fromarray(mask)
                        # PIL_mask = PIL_mask.resize(size=(nx_in,ny_in),resample=Image.NEAREST)
                        # PIL_mask.save(transfo_masks_folder+"/tmp_mask.png")
                        
                        
                        dataset_test = LakesDataset(transfo_folder, get_transform(train=False))
                        
                        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test, batch_size=1, shuffle=False, num_workers=1,
                            collate_fn=utils.collate_fn)

                        output_img_filename = output_imgs_folder+file_root+'_'+str(istart)+'_'+str(ix).zfill(2)+'_'+str(iy).zfill(2)+'.png'
                        output_poly_filename = output_poly_folder+file_root+'_'+str(istart)+'_'+str(ix).zfill(2)+'_'+str(iy).zfill(2)+'.shp'
                        
                        thresh = 0.5
                        
                        if Keep_Img_plots:
                            all_masks = get_one_mask_and_plot(model, data_loader_test, device=device,image_output_filename=output_img_filename,thresh=thresh)
                        else:
                            all_masks = get_one_mask_no_plot(model, data_loader_test, device=device,thresh=thresh)
                        
                        npoly = all_masks.shape[0]
                        print('npoly = ',npoly)

                        mask_no_col = np.zeros((nx_in,nx_in),dtype=np.uint8)
                        
                        npoly_real = 0
                        
                        eq_classes = []

                        for ipoly in range(npoly):

                            MA = torch.where(all_masks[ipoly,0,:,:] > thresh,1,0).reshape((nx_in,nx_in)).detach().cpu().numpy().astype(np.uint8)
                            
                            poly_collisions = np.where(MA,mask_no_col,0).astype(np.uint8).reshape((nx_in,nx_in))
                            
                            partial_class = set(np.unique(poly_collisions))
                            partial_class.remove(0)
                            partial_class.add(ipoly+1)
                            
                            for the_class in eq_classes:
                                
                                if partial_class.intersection(the_class):
                                    
                                    partial_class.update(the_class)
                            
                            eq_classes = [ the_class for the_class in eq_classes if not(partial_class.intersection(the_class))]
                            eq_classes.append(partial_class)
                            
                            mask_no_col = np.where(MA,(ipoly+1),mask_no_col).astype(np.uint8).reshape((nx_in,nx_in))

                        class_mask = np.zeros((nx_in,nx_in),dtype=np.uint8)
                        
                        for iclass in range(len(eq_classes)):
                            
                            for jpoly in eq_classes[iclass]:
                            
                                to_add = np.where(mask_no_col == jpoly ,(iclass+1),0).astype(np.uint8).reshape((nx_in,nx_in))
                                
                                overlap = np.sum(to_add*class_mask)
                                
                                if (overlap):
                                    raise ValueError("There was an error in merging overlapping polygons")
                                
                                class_mask += to_add
                        
                        n_poly_no_col = len(eq_classes)
                        
                        class_mask_PIL = Image.fromarray(class_mask)
                        class_mask_PIL = class_mask_PIL.resize(size=(nx_out,ny_out),resample=Image.NEAREST)
                        class_mask = np.array(class_mask_PIL).reshape((nx_out,ny_out))

                        print('npoly without collisions = ',n_poly_no_col)    
                        
                        if (n_poly_no_col > 0):

                            polyval = []
                            geometry = []
                            
                            the_win = rio.windows.Window((ystart+(iy)*ny_out),(xstart+(ix)*nx_out),ny_out,nx_out)
                            the_transform = img_open.window_transform(the_win)

                            the_polys = rio.features.shapes(class_mask, transform=the_transform)
                            
                            for shapedict, value in the_polys:

                                if (value != 0):
                                    
                                    polyval.append(str(value-1)) # ici ça commence à zéro, désolé
                                    geometry.append(shapely.geometry.shape(shapedict))
                            
                            
                            # build the gdf object over the two lists
                            gdf = gpd.GeoDataFrame(
                                {'Id': polyval, 'geometry': geometry },
                                crs=img_open.crs
                            )
                            
                            gdf.to_crs({'proj':'cea'},inplace=True) 
                            
                            gdf['Shape_Area'] = gdf.area
                            
                            gdf.to_crs(img_open.crs,inplace=True) 
                            
                            try:
                                gdf.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
                            except Exception:
                                pass
                            
            
            npoly_tot = 0
            
            Big_GDF = gpd.GeoDataFrame(
                        crs=img_open.crs
                    )

            for file_path in os.listdir(output_poly_folder):
                file_path = os.path.join(output_poly_folder, file_path)
                the_file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                
                if (file_ext == '.shp' ):
                    
                    print(file_path)
                    
                    lake_outlines = gpd.read_file(file_path)
                    
                    lake_outlines.to_crs(img_open.crs,inplace=True) 
                    
                    npoly = lake_outlines['geometry'].shape[0]
                    
                    for ipoly in range(npoly):
                        lake_outlines.loc[ipoly,'Id'] = int(float(lake_outlines.loc[ipoly,'Id'])) + npoly_tot
                        
                    npoly_tot += npoly

                    Big_GDF = gpd.GeoDataFrame( pd.concat( [Big_GDF,lake_outlines], ignore_index=True) ,crs=img_open.crs)
               
                   
            output_poly_filename = output_fused_poly_folder + 'all_polys_not_fused.shp'
            
            try:
                Big_GDF.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
            except Exception:
                pass

                   
                   
            # print(Big_GDF['geometry'][0])
            Big_GDF.to_crs({'proj':'cea'},inplace=True) 
            # print(Big_GDF['geometry'][0])

            # Big_GDF.to_crs(img_open.crs,inplace=True) 
                            
            print(npoly_tot)

            # buffer_distance = 12
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
                
                
                the_union = shapely.ops.unary_union(class_polys)
                
                if isinstance(the_union,shapely.geometry.polygon.Polygon) :
                    the_ids.append(iclass)
                    the_geometry.append(the_union)

            gdf_fused = gpd.GeoDataFrame(
                {'Id': the_ids, 'geometry': the_geometry },
                crs={'proj':'cea'}
            )
                
            gdf_fused['Shape_Area'] = gdf_fused.area
            gdf_fused.to_crs(img_open.crs,inplace=True) 

            output_poly_filename = output_fused_poly_folder + 'all_polys.shp'
            
            try:
                gdf_fused.to_file(driver = 'ESRI Shapefile', filename= output_poly_filename)
            except Exception:
                pass


            shutil.rmtree(output_poly_folder)
        
        
        
        # All model predictions done.
        # Now aggregate different predictions

        training_folders = [os.path.join(base_poly_mul_folder, o) for o in os.listdir(base_poly_mul_folder) 
                            if os.path.isdir(os.path.join(base_poly_mul_folder,o))]

        poly_img = np.zeros((nx_img,ny_img),dtype=np.uint8)
            
        print('\nReading Predictions\n')
        
        for the_training_folder in training_folders:
            
            init_lake_bou = the_training_folder+'/fused_polygons/all_polys.shp'
                
            lake_outlines = gpd.read_file(init_lake_bou)
            lake_outline_match=lake_outlines.to_crs(img_open.crs)

            npoly = lake_outline_match['geometry'].shape[0]
            print(the_training_folder,' npoly = ',npoly)

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
            
            
            print(i,'npoly = ',npoly)
            
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
