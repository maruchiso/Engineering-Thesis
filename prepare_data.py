import tifffile as tf
import numpy as np
from random import uniform as rand_value
from os import remove
from ED_Denoising.denoise_methods.BM4D import BM4D
import time




#directory = 'E:/inzynierka_dane/2017_08_29_DSP/AICS-17'
#file_list = listdir(directory) 

#od 450 do 790 "szybkie" odszumianie
#Odszumić tylko wycinek od 0 do 450

for i in range(0, 450): 

    #read the intensity of one image and cut it to the shape (55, 512, 512) 
    path1  = 'E:/inzynierka_dane/2017_08_29_DSP/AICS-17/AICS-17_'
    path3 = '.ome.tif'
    path2 = str(i)
    path = path1 + path2 + path3
    data = tf.imread(path)

    intensity = data[0:55, 3, 0:512, 0:512] #channel TL Brightfield
    

    #normalize values (0-1)
    max_intensity = np.max(intensity)
    min_intensity = np.min(intensity)
    distribution_i_normalized = (intensity - min_intensity) / (max_intensity - min_intensity)
    threshold = 4.85
    print(distribution_i_normalized.shape)
    
    try:
        start_time = time.time()
        denoiser = BM4D(distribution_i_normalized)
        denoise_data = denoiser.execute_3d(threshold)
        end_time = time.time()

        #scale to refractive index distribution
        max_n = 1.4147 + np.round(rand_value(-0.0721, 0), 4) #the biggest value of n in RECON matrix +- random small value
        min_n = 1.337 #współczynnik załamania wody
        distribution_n = np.round((((denoise_data - min_intensity) / (max_intensity - min_intensity)) * (max_n - min_n) + min_n), 4)
        
        #save to file
        outpath = 'E:/inzynierka_dane/odszumione/1crop2denoised/denoise_vol2_test' + str(i) + '.ome.tiff'
        tf.imsave(outpath, denoise_data, description=None)
        remove(path)
        exe_time = (end_time - start_time)
        print(exe_time)
    except Exception as e:
        print(f"Błąd prawdopodobnie podczas execute_3d(): {e}")