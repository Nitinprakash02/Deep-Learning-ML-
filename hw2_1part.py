#import required libraries
import urllib.request
import os
import pandas as pd
import numpy as np
import plotnine as p9

#data_url is a url from where we have to get our data
data_url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"

#path_to_file is where the data file will be stored
path_to_file = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW2/zip.test.gz"

#if data file does not exist in the system then download it else #print file already exist
if not os.path.exists(path_to_file):
    urllib.request.urlretrieve(data_url, path_to_file)

#read data from a file and make a dataframe
zip_df = pd.read_csv(path_to_file, sep = " ", header=None)

image_num = 500
n_pixels = 16

image_label = zip_df.iloc[image_num, 0]

one_img_df = pd.DataFrame({
    "row": np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
    "column": np.tile(np.arange(n_pixels),n_pixels),
    "intensity": zip_df.iloc[image_num, 1:]
})

#Visualize the data using plotnine
one_img_plot = p9.ggplot()+\
    p9.ggtitle(f"Label : {image_label}")+\
    p9.geom_tile(
        p9.aes(
            x="column",
            y="row",
            fill="intensity"
        ),
        data = one_img_df
    )+\
    p9.scale_fill_gradient(
        low="black",
        high="white"
    )
one_img_plot.save("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW2/one_img_plot")
