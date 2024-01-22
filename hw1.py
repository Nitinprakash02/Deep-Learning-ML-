#import required libraries
import urllib.request
import os
import pandas as pd

#data_url is a url from where we have to get our data
data_url = "https://github.com/tdhock/2023-08-deep-learning/raw/main/data/zip.test.gz"

#path_to_file is where the data file will be stored
path_to_file = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW1/zip.test.gz"

#if data file does not exist in the system then download it else #print file already exist
if not os.path.exists(path_to_file):
    urllib.request.urlretrieve(data_url, path_to_file)
    print("File downloaded successfully")

else:
    print("File already exists")

#read data from a file and make a dataframe
df = pd.read_csv(path_to_file, sep = " ", header=None)

#print the shape of dataframe
print("Shape of dataframe : ", df.shape)
