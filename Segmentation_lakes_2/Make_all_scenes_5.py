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
# input_img_list = [
# '/mnt/c/GeoData/Polygon_Annotations/1980_DZB1216/1980_DZB1216.tif',
# '/mnt/c/GeoData/Polygon_Annotations/1989_07_12_Spot1/SCENE01/1989_07_12_Spot1.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/1989_07_16_Spot1/SCENE01/1989_07_16_Spot1.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2005_09_25_Spot5/SCENE01/2005_09_25_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2007_08_02_Spot5/SCENE01/2007_08_02_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2011_09_11_Spot5/SCENE01/2011_09_11_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2012_07_25_Spot5/61-01_5_286221_12-07-25-0233311B0/2012_07_25_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2012_09_25_Spot5/SCENE01/2012_09_25_Spot5.TIF',
# '/mnt/c/GeoData/Polygon_Annotations/2013_07_14_Spot5/68-01_5_289222_13-07-14-0211442B0/2013_07_14_Spot5.TIF',
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
# ]

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

# input_img_list = [
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/01_5_289223_09-07-14-0226511B0_BW/SCENE01/2009_07_14_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_29a/C_1967_29a_test_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_29b/C_1967_29b_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_30/C_1967_30_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_31/C_1967_31_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_32/C_1967_32_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/C_1967_33/C_1967_33_001.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/163-01_5_285222_08-06-09-0216381B8/SCENE01/2008_06_09.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/164-01_5_286222_08-07-19-0246551B1/SCENE01/2008_07_19.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/22-01_5_286223_06-05-20-0305362B0_BW/22-01_5_286223_06-05-20-0305362B0_BW/SCENE01/2006_05_20.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/23-01_5_289223_06-06-09-0321012B0_BW/23-01_5_289223_06-06-09-0321012B0_BW/SCENE01/2006_06_09.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/30-01_5_286223_07-06-21-0228362B0_BW/30-01_5_286223_07-06-21-0228362B0_BW/SCENE01/2007_06_21.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/31-01_5_289223_07-07-02-0216501B0_BW/31-01_5_289223_07-07-02-0216501B0_BW/SCENE01/2007_07_02.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/32-01_5_285222_07-07-04-0318302B0/SCENE01/2007_07_04.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/39-01_5_285221_08-07-24-0250351B0/SCENE01/2008_07_24_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/40-01_5_285222_08-07-24-0250431B0/SCENE01/2008_07_24_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/41-01_5_286222_08-09-09-0245551B0/SCENE01/2008_09_09_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/42-01_5_286223_08-09-09-0246031B0/SCENE01/2008_09_09_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/43-01_5_286220_10-10-03-0254522B0_BW/43-01_5_286220_10-10-03-0254522B0_BW/SCENE01/2010_10_03_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/44-01_5_286221_10-10-03-0255002B0/SCENE01/2010_10_03_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/46-01_5_286223_10-10-03-0255162B0_BW/46-01_5_286223_10-10-03-0255162B0_BW/SCENE01/2010_10_03_003.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/5-01_5_289222_09-07-14-0226441B0_BW/SCENE01/2009_07_14_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/53-01_5_286220_11-08-26-0304132B0_BW/53-01_5_286220_11-08-26-0304132B0_BW/SCENE01/2011_08_26.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/62-01_5_286222_12-07-25-0233381B0_BW/62-01_5_286222_12-07-25-0233381B0_BW/SCENE01/2012_07_25_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/63-01_5_286223_12-07-25-0233461B0_BW/63-01_5_286223_12-07-25-0233461B0_BW/SCENE01/2012_07_25_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/69-01_5_289223_13-07-19-0215282B0/69-01_5_289223_13-07-19-0215282B0/SCENE01/2013_07_19.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/79-01_5_285222_13-09-02-0247482B0_BW/79-01_5_285222_13-09-02-0247482B0_BW/SCENE01/2013_09_02.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_29a/C_1967_29a_test_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_29b/C_1967_29b_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_30/C_1967_30_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_31/C_1967_31_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_32/C_1967_32_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/C_1967_33/C_1967_33_002.tif',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/UniversiteParisSud_A.SEJOURNE_SO20004818-5-01_52892220907140226441B0/SCENE01/2009_07_14_003.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/UniversiteParisSud_A.SEJOURNE_SO20004818-6-01_52892230907140226511B0/SCENE01/2009_07_14_004.TIF',
# ]
# 
# 
# 
#   
# input_img_list = [
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/55-01_5_289223_11-09-02-0229432B0_BW/55-01_5_289223_11-09-02-0229432B0_BW/SCENE01/2011_09_02.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/56-01_5_289222_11-09-08-0213592B0_BW/56-01_5_289222_11-09-08-0213592B0_BW/SCENE01/2011_09_08.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/58-01_5_286221_11-09-21-0303512B0/SCENE01/2011-09-21.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/59-01_5_286222_11-09-21-0303592B0_BW/59-01_5_286222_11-09-21-0303592B0/SCENE01/2011_09_21.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/65-01_5_289221_12-09-25-0239152B_BW/65-01_5_289221_12-09-25-0239152B0/SCENE01/2012_09_25_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/66-01_5_289222_12-09-25-0239232B0/66-01_5_289222_12-09-25-0239232B0/SCENE01/2012_09_25_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/67-01_5_289223_12-09-25-0239312B0/67-01_5_289223_12-09-25-0239312B0/SCENE01/2012_09_25_003.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/71-01_5_286221_13-08-19-0217252B0/71-01_5_286221_13-08-19-0217252B0/SCENE01/2013_08_19.TIF',
# ]  

# input_img_list = [
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/131-01_4_286221_10-09-26-0238301M0/131-01_4_286221_10-09-26-0238301M0/42862211009260238301M0/2010_09_26_001.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/133-01_4_286223_10-09-26-0238461M0/133-01_4_286223_10-09-26-0238461M0/SCENE01/2010_09_26_002.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/142-01_1285222_89-07-16-0319441P0/UniversiteParisSud_A.SEJOURNE_SO20004818-142-01_12852228907160319441P0/SCENE01/1989_07_16.TIF',
# '/mnt/c/GeoData/New_scenes/Scenes_BW_New/143-01_4_289220_10-08-21-0231452M7/143-01_4_289220_10-08-21-0231452M7/SCENE01/2010_08_21.TIF',
# ]

input_img_list = [
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/84-01_5_286223_07-06-21-0228362B0_color/84-01_5_286223_07-06-21-0228362B0/DATA_N1915.17_E146.32/SPVIEW__2020_0/2007_06_21_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/84-01_5_286223_07-06-21-0228362B0_color/84-01_5_286223_07-06-21-0228362B0/DATA_N1915.17_E160.21/SPVIEW__2020_0/2007_06_21_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/88-01_5_289223_11-09-02-0229432B0_color/88-01_5_289223_11-09-02-0229432B0_color/DATA_N1915.33_E155.92/SPVIEW__2020_0/2011_09_02_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/88-01_5_289223_11-09-02-0229432B0_color/88-01_5_289223_11-09-02-0229432B0_color/DATA_N1915.33_E169.81/SPVIEW__2020_0/2011_09_02_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/89-01_5_289222_11-09-08-0213592B0_color/89-01_5_289222_11-09-08-0213592B0_color/DATA_N1929.06_E160.61/SPVIEW__2020_0/2011_09_08_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/89-01_5_289222_11-09-08-0213592B0_color/89-01_5_289222_11-09-08-0213592B0_color/DATA_N1929.06_E174.50/SPVIEW__2020_0/2011_09_08_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/90-01_5_286222_11-09-11-0256292B0_color/90-01_5_286222_11-09-11-0256292B0_color/DATA_N1929.15_E147.24/SPVIEW__2020_0/2011_09_11_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/90-01_5_286222_11-09-11-0256292B0_color/90-01_5_286222_11-09-11-0256292B0_color/DATA_N1929.15_E161.12/SPVIEW__2020_0/2011_09_11_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/91-01_5_286221_11-09-21-0303512B0_color/91-01_5_286221_11-09-21-0303512B0_color/DATA_N1943.14_E152.28/SPVIEW__2020_0/2011_09_21_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/91-01_5_286221_11-09-21-0303512B0_color/91-01_5_286221_11-09-21-0303512B0_color/DATA_N1943.14_E166.17/SPVIEW__2020_0/2011_09_21_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/92-01_5_286222_11-09-21-0303592B0_color/92-01_5_286222_11-09-21-0303592B0_color/DATA_N1929.40_E146.67/SPVIEW__2020_0/2011_09_21_003.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/92-01_5_286222_11-09-21-0303592B0_color/92-01_5_286222_11-09-21-0303592B0_color/DATA_N1929.40_E160.56/SPVIEW__2020_0/2011_09_21_004.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/93-01_5_289221_12-09-25-0239152B0_color/93-01_5_289221_12-09-25-0239152B0_color/DATA_N1943.24_E167.57/SPVIEW__2020_0/2012_09_25_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/93-01_5_289221_12-09-25-0239152B0_color/93-01_5_289221_12-09-25-0239152B0_color/DATA_N1943.24_E181.46/SPVIEW__2020_0/2012_09_25_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/94-01_5_289222_12-09-25-0239232B0_color/94-01_5_289222_12-09-25-0239232B0_color/DATA_N1929.44_E162.88/SPVIEW__2020_0/2012_09_25_003.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/94-01_5_289222_12-09-25-0239232B0_color/94-01_5_289222_12-09-25-0239232B0_color/DATA_N1929.44_E176.77/SPVIEW__2020_0/2012_09_25_004.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/95-01_5_289223_12-09-25-0239312B0_color/95-01_5_289223_12-09-25-0239312B0_color/DATA_N1915.64_E158.16/SPVIEW__2020_0/2012_09_25_005.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/95-01_5_289223_12-09-25-0239312B0_color/95-01_5_289223_12-09-25-0239312B0_color/DATA_N1915.64_E172.05/SPVIEW__2020_0/2012_09_25_006.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/96-01_5_286221_13-08-19-0217252B0_color/96-01_5_286221_13-08-19-0217252B0_color/DATA_N1942.59_E151.33/SPVIEW__2020_0/2013_08_19.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/96-01_5_286221_13-08-19-0217252B0_color/96-01_5_286221_13-08-19-0217252B0_color/DATA_N1942.59_E165.22/SPVIEW__2020_0/IMAGERY.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/97-01_5_286221_13-08-24-0221032B0_color/97-01_5_286221_13-08-24-0221032B0_color/DATA_N1942.64_E151.60/SPVIEW__2020_0/2013_08_24_001.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/97-01_5_286221_13-08-24-0221032B0_color/97-01_5_286221_13-08-24-0221032B0_color/DATA_N1942.64_E165.49/SPVIEW__2020_0/2013_08_24_002.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906160223161_ORT_SPOT7_20220414_1249371xpqav8ihwwo9_1_R1C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906160223161_ORT_SPOT7_20220414_1249371xpqav8ihwwo9_1_R1C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906160223161_ORT_SPOT7_20220414_1249371xpqav8ihwwo9_1_R2C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_133/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906160223161_ORT_SPOT7_20220414_1249371xpqav8ihwwo9_1_R2C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R1C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R1C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R2C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R2C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R3C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_134/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906170217486_ORT_SPOT6_20220414_1250221d1dddrhqa7iq_1_R3C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906180209561_ORT_SPOT7_20220414_1251471izznmsxga4fv_1_R1C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906180209561_ORT_SPOT7_20220414_1251471izznmsxga4fv_1_R1C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906180209561_ORT_SPOT7_20220414_1251471izznmsxga4fv_1_R2C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_135/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/IMG_SPOT7_MS_201906180209561_ORT_SPOT7_20220414_1251471izznmsxga4fv_1_R2C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906190204145_ORT_SPOT6_20220414_1252311a8d226q5vds4_1_R1C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906190204145_ORT_SPOT6_20220414_1252311a8d226q5vds4_1_R1C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906190204145_ORT_SPOT6_20220414_1252311a8d226q5vds4_1_R2C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_136/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201906190204145_ORT_SPOT6_20220414_1252311a8d226q5vds4_1_R2C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R1C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R1C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R2C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R2C2.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R3C1.TIF',
'/mnt/c/GeoData/New_scenes/Scenes_Color_New/SPOT6_2019_HC-Ortho_NC_DRS-MS_SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/SPOT6_2019_HC_ORTHO_NC_GEOSUD_MS_137/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201907130218426_ORT_SPOT6_20220414_1255161uhbygy74saku_1_R3C2.TIF',
]
    

for file_path in input_img_list:
    if not(os.path.isfile(file_path)):
        raise ValueError(f'This is not a valid file : {file_path}')



target_mean_list = [    120 ,120,120,140 ,140,140,160 ,160,160,]
target_stddev_list = [  40  ,60 ,80 ,40  ,60 ,80 ,40  ,60 ,80 ,]

# target_mean_list = [   140]
# target_stddev_list = [  60]

# target_mean_list = [   None]
# target_stddev_list = [  None]

n_contrast = len(target_mean_list)

Local_contrast_list = [False for i in range(n_contrast)]
# Local_contrast_list = [True for i in range(n_contrast)]


for contrast_params in [target_mean_list,target_stddev_list,Local_contrast_list]:
    if not(len(contrast_params) == n_contrast):
        raise ValueError('INVALID CONTRAST CHANGE DEFINITION')




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


Additional_filters = []
# Additional_filters = [ImageFilter.BLUR]
# Additional_filters = [ImageFilter.SMOOTH_MORE]


# model_list = [
# "./trainings/02_SGD_resnet18/MyTraining_001.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_000.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_001.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_002.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_003.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_004.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_005.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_006.pt",
# "./trainings/04_SGD_resnet152_before_second_bug/MyTraining_007.pt",
# ]

model_list = [
"./trainings/05_SGD_resnet152_final/MyTraining_005.pt",
]





# Keep_Img_plots = True
Keep_Img_plots = False

mod_img_lara_fix = True
# mod_img_lara_fix = False

n_input_img = len(input_img_list)
n_model = len(model_list)


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

        
        for i_contrast in range(n_contrast):
            
            target_mean = target_mean_list[i_contrast]
            target_stddev = target_stddev_list[i_contrast]
            Local_contrast = Local_contrast_list[i_contrast]

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
                
            else:
                
                nskew = 0
                
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



            for i_model in range(n_model):
                the_model = model_list[i_model]
                
                the_basename = os.path.basename(the_model)
                model_root,ext = os.path.splitext(the_basename)
                
                base_model_folder =base_poly_mul_folder+model_root+'_'+str(i_contrast).zfill(2)+'/'

                output_poly_folder = base_model_folder +'polygons/'
                output_fused_poly_folder = base_model_folder+'fused_polygons/'
                output_imgs_folder = base_model_folder+'images/'

                for store_folder in [transfo_folder,transfo_masks_folder,transfo_imgs_folder,output_imgs_folder,output_poly_folder,output_fused_poly_folder]:
                    if not(os.path.isdir(store_folder)):
                        os.makedirs(store_folder)

                filename_path_save = base_model_folder+'Contrast_settings.txt'

                with open(filename_path_save, 'w') as outfile:
                    outfile.write(f'target_mean = {target_mean}\n')
                    outfile.write(f'target_stddev = {target_stddev}\n')
                    outfile.write(f'Local_contrast = {Local_contrast}\n')


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
                            

                            # print(model_root,' ',i_contrast+1,n_contrast,istart+1,istartmax,ix+1,ixmax,iy+1,iymax)
                            # print(i_img,'/',n_input_img,i_contrast+1,n_contrast,istart+1,istartmax,ix+1,ixmax,iy+1,iymax)

                            info_msg = [
                                (i_img,n_input_img),
                                (i_contrast+1,n_contrast),
                                (i_model+1,n_model),
                                (istart+1,istartmax),
                                (ix+1,ixmax),
                                (iy+1,iymax),
                            ]
                            
                            printlist = [f'{i}/{j}   ' for (i,j) in info_msg]
                            printmsg = ''.join(printlist)
                            print(printmsg)

                            # img_uint8_small = img_uint8[xstart+ix*nx_out:xstart+(ix+1)*nx_out,ystart+iy*ny_out:ystart+(iy+1)*ny_out]
                            
                            xi = xstart+ix*nx_out
                            xf = xstart+(ix+1)*nx_out
                            yi = ystart+iy*ny_out
                            yf = ystart+(iy+1)*ny_out
                            img_uint8_small = np_safe_copy(img_uint8,xi,xf,yi,yf)
                            
                            if Local_contrast and not((target_stddev is None) and (target_mean is None)):

                                # nskew = 10
                                nskew = 0

                               
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
                                
                            
                            for filer in Additional_filters:
                            
                                PIL_img = PIL_img.filter(filer)
                            
                            
                            PIL_img.save(transfo_imgs_folder+"/tmp_img.png")
                            
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
                    # print('buffer',ipoly,npoly_tot)
                    
                    if (buffer_distance == 0):
                        exp_geometry.append(Big_GDF['geometry'][ipoly])
                    else:
                        exp_geometry.append(Big_GDF['geometry'][ipoly].buffer(buffer_distance))
                    
                inter_ij = np.full((npoly_tot,npoly_tot),False,dtype=bool)

                print('Computing intersections')

                for ipoly in range(npoly_tot):
                    # print('intersection',ipoly,npoly_tot)
                    inter_ij[ipoly,ipoly] = True
                    for jpoly in range(ipoly+1,npoly_tot):
                        inter_ij[ipoly,jpoly] = exp_geometry[ipoly].intersects(exp_geometry[jpoly])
                        inter_ij[jpoly,ipoly] = inter_ij[ipoly,jpoly]
                
                print('Building Classes')
                
                eq_classes = []
                for ipoly in range(npoly_tot):
                    # print('class',ipoly,npoly_tot)
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
                    # print('fuse',iclass,len(eq_classes))
                    
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
