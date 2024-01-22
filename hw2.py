#import required libraries
import urllib.request
import os
import pandas as pd
import numpy as np
import plotnine as p9
from math import sqrt

#data_url is a url from where we have to get our data
data_url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"

#path_to_file is where the data file will be stored
path_to_file = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW2/zip.test.gz"

#if data file does not exist in the system then download it else #print file already exist
if not os.path.exists(path_to_file):
    urllib.request.urlretrieve(data_url, path_to_file)

#read data from a file and make a dataframe
zip_df = pd.read_csv(path_to_file, sep = " ", header=None)

#print the shape of dataframe
print("Shape of dataframe : ", zip_df.shape)

list_of_img_frame = []

for img_num in [5, 6, 7, 8, 9, 10, 11, 15, 20]:
    one_img_label = zip_df.iloc[img_num, 0]
    intensity_vector = zip_df.iloc[img_num, 1:]
    n_pixels = int(sqrt(len(intensity_vector)))
    one_img_df = pd.DataFrame({
        "observation": img_num,
        "label": one_img_label,
        "intensity": intensity_vector,
        "row": np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
        "column": np.tile(np.arange(n_pixels),n_pixels)
    })
    list_of_img_frame.append(one_img_df)

several_img_df = pd.concat(list_of_img_frame)

#Visualize the data using plotnine
several_img_plot = p9.ggplot()+\
    p9.facet_wrap(["observation", "label"], labeller="label_both")+\
    p9.geom_tile(
        p9.aes(
            x="column",
            y="row",
            fill="intensity"
        ),
        data = several_img_df
    )+\
    p9.scale_fill_gradient(
        low="black",
        high="white"
    )


several_img_plot.save("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW2/several_img_plot.png")
