import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import leafmap
from samgeo import tms_to_geotiff
from samgeo.text_sam import LangSAM

import rasterio
from rasterio.plot import show

# Data Paths
ROOT_IMAGE_DIR = '/Users/riaz/Desktop/images/'
ROOT_MASK_DIR = '/Users/riaz/Desktop/masks/'
METADATA_CSV_PATH = '/Users/riaz/Desktop/meta_data.csv'
TEST_FOREST_DIR = '/Users/riaz/Desktop/test_forest_2/'
OUTPUT_DIR = '/Users/riaz/Desktop/output/'

sam = LangSAM()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def look_all_files(folder_path, path_to_save):
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            FILENAME = os.path.join(root, filename)
            file_name = os.path.splitext(filename)[0]
            text_prompt = 'forest'
#             img = rasterio.open(FILENAME)
#             show(img.read(3))
            sam.predict(FILENAME,
                        text_prompt,
                        box_threshold=0.24,
                        text_threshold=0.24)
            sam.show_anns(
                cmap='Greens',
                add_boxes=False,
                alpha=0.5,
                title='Tree Region Segmentation',
                output = path_to_save + file_name + 'output.tif'
            )
#             output = Image.open('output.tif')
#             output.show()            
            print("-"*20)


start = time.time()
look_all_files(TEST_FOREST_DIR, OUTPUT_DIR)
end = time.time()

print("Time taken: ", end-start)