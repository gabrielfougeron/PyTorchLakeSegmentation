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


import warnings

import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import *
import utils
import transforms as T

from tv_training_code import *

device = torch.device('cuda')


# input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Scenes for Analysis/44-01_5_286221_10-10-03-0255002B0/SCENE01'
input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/'
# input_image_folder = '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/'


# target_mean = 127
target_mean = 120
target_stddev = 40


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
"./trainings/5/MyTraining_019.pt",
"./trainings/5/MyTraining_028.pt",
"./trainings/5/MyTraining_038.pt",
"./trainings/5/MyTraining_047.pt",
"./trainings/5/MyTraining_057.pt",
"./trainings/5/MyTraining_085.pt",
]


input_img_list = []
for file_path in os.listdir(input_image_folder):
    file_p = os.path.join(input_image_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_p))
    
    if (file_ext.lower() == '.tif' ):
        input_img_list.append(file_path)
        
        
print(input_img_list)
        
input_img_list = [input_img_list[0]]

n_input_img = len(input_img_list)

# base_base_folder = './polygons_mul/'
base_base_folder = './polygons_mul_stash/'


for the_model in model_list:
    
    the_basename = os.path.basename(the_model)
    root,ext = os.path.splitext(the_basename)
    
    print(root,ext)
    
    model = torch.load(the_model)

    transfo_folder = './transformed_make_scenes/'
    transfo_masks_folder = transfo_folder+'Lakes_masks/'
    transfo_imgs_folder = transfo_folder+'Lakes_png_images/'

    base_folder =base_base_folder+root+'/'

    output_poly_folder = base_folder +'polygons/'
    output_fused_poly_folder = base_folder+'fused_polygons/'
    output_imgs_folder = base_folder+'images/'

    for store_folder in [transfo_folder,transfo_masks_folder,transfo_imgs_folder,output_imgs_folder,output_poly_folder,output_fused_poly_folder]:
        if not(os.path.isdir(store_folder)):
            os.makedirs(store_folder)


    model.to(device)

    i_img = 0

    for file_path in input_img_list:
        file_p = os.path.join(input_image_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_p))
        
        poly_filename = output_poly_folder+file_root+'.shp'
            
        i_img = i_img + 1
        print('')
        print('----------------------------------------------')
        print('')   
        print('Processing image '+str(i_img)+" of "+str(n_input_img))
        print(file_root)
        

        # Open raster image
        with rio.open(file_p) as img_open:
            img = img_open.read()
            
            vals, count = np.unique(img , return_counts=True)
            
            vals = vals[1:]
            count = count[1:]
            
            mean = np.sum(vals*count)/np.sum(count)
            stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
            
            img_new = ((img.astype(np.float32) - mean) * (target_stddev/stddev) + target_mean)
            
            img_new = np.where(img_new > 255.,255.,img_new)
            img_new = np.where(img_new < 0.,0.,img_new)
            img_new = img_new.astype(np.uint8)
            
            img_uint8 = np.where(img[0,:,:] == 0,np.uint8(0),img_new[0,:,:])

            # vals, count = np.unique(img , return_counts=True)
            
            # vals = vals[1:]
            # count = count[1:]
            
            # mean = np.sum(vals*count)/np.sum(count)
            # stddev = np.sqrt(np.sum(((vals-mean)**2)*count)/np.sum(count))
   
            # print(mean,stddev)
            
            # fig, ax = plt.subplots()
            # plt.bar(vals, count)
            # plt.savefig("bar.png")


            del img
            del img_new

            print("image loaded")
            
            
            nxtot = img_uint8.shape[0]
            nytot = img_uint8.shape[1]
            print('nxtot = ',nxtot)
            print('nytot = ',nytot)
                
            BB = plotting_extent(img_open)
            xmin,xmax,ymin,ymax = plotting_extent(img_open)
            # xmin,ymin,xmax,ymax = img_open.bounds
            print('xmin = ',xmin)
            print('xmax = ',xmax)
            print('ymin = ',ymin)
            print('ymax = ',ymax)

            
            xstart_list = [0,nx_out//2]
            ystart_list = [0,ny_out//2]
            
            istartmin = 0
            ixmin = 0
            iymin = 0
            
            
            for istart in range(istartmin,len(xstart_list)):
                
                xstart = xstart_list[istart]
                ystart = ystart_list[istart]

                ixmax = (nxtot-xstart)//nx_out
                iymax = (nytot-ystart)//ny_out
            
                for ix in range(ixmin,ixmax):
                    for iy in range(iymin,iymax):
                        
                        print(istart,ix,ixmax,iy,iymax)
                
                
                        
                        # img_uint8_small = img_uint8[xstart+ix*nx_out:xstart+(ix+1)*nx_out,ystart+iy*ny_out:ystart+(iy+1)*ny_out]
                        
                        # PIL_img = Image.fromarray(img_uint8_small)
                        # PIL_img = PIL_img.resize(size=(nx_in,ny_in),resample=Image.BICUBIC)
                        # PIL_img.save(transfo_imgs_folder+"/img_.png")
                
            
                        img_uint8_small = img_uint8[xstart+ix*nx_out:xstart+(ix+1)*nx_out,ystart+iy*ny_out:ystart+(iy+1)*ny_out]
                        PIL_img = Image.fromarray(img_uint8_small)
                        PIL_img = PIL_img.resize(size=(nx_in,ny_in),resample=Image.BICUBIC)
                        PIL_img.save(transfo_imgs_folder+"/tmp_img.png")
                        
                        mask = np.zeros((nx_out,ny_out),dtype=np.uint8)
                        mask[0:(nx_out//2),0:(ny_out//2)]=1
                        PIL_mask = Image.fromarray(mask)
                        PIL_mask = PIL_mask.resize(size=(nx_in,ny_in),resample=Image.NEAREST)
                        PIL_mask.save(transfo_masks_folder+"/tmp_mask.png")
                        
                        
                        dataset_test = LakesDataset(transfo_folder, get_transform(train=False))
                        
                        data_loader_test = torch.utils.data.DataLoader(
                            dataset_test, batch_size=1, shuffle=False, num_workers=1,
                            collate_fn=utils.collate_fn)

                        output_img_filename = output_imgs_folder+file_root+'_'+str(istart)+'_'+str(ix).zfill(2)+'_'+str(iy).zfill(2)+'.png'
                        output_poly_filename = output_poly_folder+file_root+'_'+str(istart)+'_'+str(ix).zfill(2)+'_'+str(iy).zfill(2)+'.shp'
                        
                        thresh = 0.5
                        
                        all_masks = get_one_mask_and_plot(model, data_loader_test, device=device,image_output_filename=output_img_filename,thresh=thresh)
                        # all_masks = get_one_mask_no_plot(model, data_loader_test, device=device,thresh=thresh)
                        
                        npoly = all_masks.shape[0]
                        print('npoly = ',npoly)
                        
                        mask_no_col = np.zeros((nx_out,ny_out),dtype=np.uint8)
                        
                        npoly_real = 0
                        
                        eq_classes = []
                        # poly_to_eq_class = []
                        # n_eq_class = 0
                        
                        for ipoly in range(npoly):
                            
                            # MA = torch.where(all_masks[ipoly,:,:] > thresh,1,0).reshape((nx_out,ny_out)).detach().cpu().numpy().astype(np.uint8)
                            
                            all_masks_ipoly = np.copy(all_masks[ipoly,0,:,:].detach().cpu().numpy())
                            all_masks_ipoly_PIL = Image.fromarray(all_masks_ipoly)
                            all_masks_ipoly_PIL = all_masks_ipoly_PIL.resize(size=(nx_out,ny_out),resample=Image.BICUBIC)
                            all_masks_ipoly = torch.from_numpy(np.array(all_masks_ipoly_PIL).reshape((1,nx_out,ny_out)))
                            MA = torch.where(all_masks_ipoly > thresh,1,0).reshape((nx_out,ny_out)).detach().cpu().numpy().astype(np.uint8)
                            
                            poly_collisions = np.where(MA,mask_no_col,0).astype(np.uint8).reshape((nx_out,ny_out))
                            
                            partial_class = set(np.unique(poly_collisions))
                            partial_class.remove(0)
                            partial_class.add(ipoly+1)
                            
                            for the_class in eq_classes:
                                
                                if partial_class.intersection(the_class):
                                    
                                    partial_class.update(the_class)
                            
                            eq_classes = [ the_class for the_class in eq_classes if not(partial_class.intersection(the_class))]
                            eq_classes.append(partial_class)
                            
                            mask_no_col = np.where(MA,(ipoly+1),mask_no_col).astype(np.uint8).reshape((nx_out,ny_out))

                        class_mask = np.zeros((nx_out,ny_out),dtype=np.uint8)
                        
                        for iclass in range(len(eq_classes)):
                            
                            for jpoly in eq_classes[iclass]:
                            
                                to_add = np.where(mask_no_col == jpoly ,(iclass+1),0).astype(np.uint8).reshape((nx_out,ny_out))
                                
                                overlap = np.sum(to_add*class_mask)
                                
                                if (overlap):
                                    raise ValueError("There was an error in merging overlapping polygons")
                                
                                class_mask += to_add
                        
                        n_poly_no_col = len(eq_classes)
                        
                        print('npoly without collisions = ',n_poly_no_col)    
                        
                        if (n_poly_no_col > 0):

                            polyval = []
                            geometry = []
                            
                            the_win = rio.windows.Window((ystart+(iy)*ny_out),(xstart+(ix)*nx_out),ny_out,nx_out)
                            the_transform = img_open.window_transform(the_win)
            
                            # the_polys = rio.features.shapes(class_mask, transform=img_open.transform)
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






                
                        
print('')
print('Done !')
