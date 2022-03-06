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


# input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
# ]


# ALL WORKING IMAGES
# input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1980_DZB1216/1980_DZB1216.tif',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1989_07_12_Spot1/SCENE01/1989_07_12_Spot1.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1989_07_16_Spot1/SCENE01/1989_07_16_Spot1.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/2005_09_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2007_08_02_Spot5/SCENE01/2007_08_02_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_23_Spot4/SCENE01/2010_09_23_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4/130-01_4_286220_10-09-26-0238221M0/SCENE01/2010-09-26_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4c/SCENE01/2010_09_26_Spot4c.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2011_09_11_Spot5/SCENE01/2011_09_11_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_23_Spot5/SCENE01/2013_08_23_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_23_Spot5b/SCENE01/2013_08_23_Spot5b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_24_Spot5/SCENE01/2013_08_24_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_24_Spot5b/SCENE01/2013_08_24_Spot5b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2016_Spot7/Spot7_Syrdakh_BW.tif',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/1989_07_16_Spot1b/SCENE01/1989_07_16_Spot1b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2011_07_30_Spot4/SCENE01/2011_07_30_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_08_24_Spot5c/SCENE01/2013_08_24_Spot5c.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_08_30_Spot5/SCENE01/2013_08_30_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_09_02_Spot5/SCENE01/2013_09_02_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
# '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
# ]






# List of images that DID NOT work :


# List of small images => nx_out = 1024


# input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_23_Spot4/SCENE01/2010_09_23_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4/130-01_4_286220_10-09-26-0238221M0/SCENE01/2010-09-26_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2010_09_26_Spot4c/SCENE01/2010_09_26_Spot4c.TIF',
]

# List of big images => nx_out = 2048


# input_img_list = [
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1980_DZB1216/1980_DZB1216.tif',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1989_07_12_Spot1/SCENE01/1989_07_12_Spot1.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/1989_07_16_Spot1/SCENE01/1989_07_16_Spot1.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/2005_09_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2007_08_02_Spot5/SCENE01/2007_08_02_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2011_09_11_Spot5/SCENE01/2011_09_11_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_23_Spot5/SCENE01/2013_08_23_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_23_Spot5b/SCENE01/2013_08_23_Spot5b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_24_Spot5/SCENE01/2013_08_24_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2013_08_24_Spot5b/SCENE01/2013_08_24_Spot5b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/2016_Spot7/Spot7_Syrdakh_BW.tif',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/1989_07_16_Spot1b/SCENE01/1989_07_16_Spot1b.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2011_07_30_Spot4/SCENE01/2011_07_30_Spot4.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_08_24_Spot5c/SCENE01/2013_08_24_Spot5c.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_08_30_Spot5/SCENE01/2013_08_30_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/Polygon_Annotations/extra/2013_09_02_Spot5/SCENE01/2013_09_02_Spot5.TIF',
# '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_N/2010_10_03N0.TIF',
# '/mnt/c/Users/Gabriel/GeoData/training_gab_01_23_2022/2010_10_03_S/Spot2_2010_10_03b0.TIF',
# ]




for file_path in input_img_list:
    if not(os.path.isfile(file_path)):
        raise ValueError(f'This is not a valid file : {file_path}')



# target_mean = 127
target_mean = 120
target_stddev = 40


# nx_out = 4096
# ny_out = 4096

# nx_out = 2048
# ny_out = 2048

nx_out = 1024
ny_out = 1024

# nx_out = 512
# ny_out = 512


    
# nx_in = nx_out
# ny_in = ny_out

    
nx_in = 1024
ny_in = 1024

model_list = [
# "./trainings/5/MyTraining_019.pt",
# "./trainings/5/MyTraining_028.pt",
# "./trainings/5/MyTraining_038.pt",
"./trainings/5/MyTraining_047.pt",
# "./trainings/5/MyTraining_057.pt",
# "./trainings/5/MyTraining_085.pt",
]

Keep_Img_plots = True
# Keep_Img_plots = False

n_input_img = len(input_img_list)



all_pred_folder = './all_predictions/'

transfo_folder = './transformed_make_scenes/'
transfo_masks_folder = transfo_folder+'Lakes_masks/'
transfo_imgs_folder = transfo_folder+'Lakes_png_images/'


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

        base_img_folder = all_pred_folder+file_root+'/'
        base_poly_mul_folder = base_img_folder+'/polygons_mul/'
        output_aggreg_poly_folder = base_img_folder+'/polygons_mul_fused/'

        for store_folder in [base_img_folder,base_poly_mul_folder,output_aggreg_poly_folder]:
            if not(os.path.isdir(store_folder)):
                os.makedirs(store_folder)
        
        filename_path_save = base_img_folder+'img_path.txt'
        
        if os.path.isfile(filename_path_save):
            raise ValueError(f'File {filename_path_save} already exists')
        
        with open(filename_path_save, 'w') as outfile:
            outfile.write(file_path)

        for the_model in model_list:
            
            the_basename = os.path.basename(the_model)
            root,ext = os.path.splitext(the_basename)
            
            print(root,ext)
            
            base_model_folder =base_poly_mul_folder+root+'/'

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

            # xstart_list = [0,nx_out//2]
            # ystart_list = [0,ny_out//2]

            xstart_list = [0]
            ystart_list = [0]
            
            istartmin = 0
            istartmax = len(xstart_list)

            for istart in range(istartmin,istartmax):
                
                xstart = xstart_list[istart]
                ystart = ystart_list[istart]
                
            
                ixmin = (nxtot-xstart)//(2*nx_out)
                iymin = (nytot-ystart)//(2*ny_out)
                

                ixmax = ixmin+1
                iymax = iymin+1

                for ix in range(ixmin,ixmax):
                    for iy in range(iymin,iymax):
                        
                        print(istart,istartmax,ix,ixmax,iy,iymax)

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
                        
                        if Keep_Img_plots:
                            all_masks = get_one_mask_and_plot(model, data_loader_test, device=device,image_output_filename=output_img_filename,thresh=thresh)
                        else:
                            all_masks = get_one_mask_no_plot(model, data_loader_test, device=device,thresh=thresh)
                        
                        npoly = all_masks.shape[0]
                        print('npoly = ',npoly)
                        
                        mask_no_col = np.zeros((nx_out,ny_out),dtype=np.uint8)
                        
                        npoly_real = 0
                        
                        eq_classes = []

                        for ipoly in range(npoly):

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
            print('npoly = ',npoly)

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
            
            
            print('npoly = ',npoly)
            
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
