import tifffile as tf
import numpy as np
#import cv2 as cv
import pandas as pd
from random import uniform as rand_value
from os import listdir
import csv

#tylko jeden katalog
directory = 'C:/Users/szymc/Desktop/inzynierka/odszumianie/2017_03_08_Struct_First_Pass_Seg/AICS-17/'
file_list = listdir(directory)

for i in range(len(file_list)):
    #z czytywanie intensywności z pojedynczego zdjęcia .ome.tiff
    path1  = 'C:/Users/szymc/Desktop/inzynierka/odszumianie/2017_03_08_Struct_First_Pass_Seg/AICS-17/AICS-17_'
    path3 = '.ome.tif'
    path2 = str(i)
    path = path1 + path2 + path3
    data = tf.imread(path)

    intensity = data[:, 3, :, :] #channel TL Brightfield

    #przeskalowanie intensywności na współczynnik załamania
    max_intensity = np.max(intensity)
    min_intensity = np.min(intensity)
    max_n = 1.4147 + np.round(rand_value(-0.0721, 0), 4) #the biggest value of n in RECON matrix +- random small value
    min_n = 1.337 #współczynnik załamania wody

    distribution_n = np.round((((intensity - min_intensity) / (max_intensity - min_intensity)) * (max_n - min_n) + min_n), 4)
    
    outpath = 'C:/Users/szymc/Desktop/inzynierka/data/data_frame_' + str(i) + '.txt'
    #save distribution index to file 
    '''
    for j in range(distribution_n.shape[0]):
        distribution_to_file = pd.DataFrame(distribution_n[j].reshape(distribution_n.shape[0], -1))
        distribution_to_file.to_csv(outpath, sep=',', index=False)
    '''
    #distribution_n_flat = distribution_n.reshape(56, -1)
    #df = pd.DataFrame(distribution_n_flat)
    #df.to_csv(outpath)
    print(distribution_n)
    