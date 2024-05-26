from tifffile import imwrite, imread
import numpy as np
from random import uniform 
import time

for i in range(0, 800): 
    try:
        start_time = time.time()
        path1  = 'E:/inzynierka_dane/desmoplakin/'
        path3 = '.ome.tif'
        path2 = str(i)
        path = path1 + path2 + path3
        data = imread(path)

        intensity = data[0:32, 3, 0:512, 0:512] #channel TL Brightfield

        #scale to refractive index distribution
        max_n = 1.4147 + np.round(uniform(-0.0150, 0), 4) #the biggest value of n in RECON matrix +- random small value
        min_n = 1.337 #distribution index of water
        distribution_n = np.round((((intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))) * (max_n - min_n) + min_n), 4)
        print(np.max(distribution_n), np.min(distribution_n))
        #save to file
        outpath = 'E:/inzynierka_dane/data/X_resized/' + str(i) + '.ome.tiff'
        imwrite(outpath, distribution_n, description=None)
        end_time = time.time()
        exe_time = (end_time - start_time)
        print(exe_time)
    except Exception as e:
        print(e)

