import random
from tifffile import imread, imwrite
import numpy as np
import bm3d

def noise(shape, mean=0, std=1):
    return np.random.normal(mean, std, shape)

def add_noise(img):
    rand_number = random.uniform(0.0, 0.25)
    noise = noise(img.shape, std=round(rand_number, 4))
    noisy_img1 = img + img * noise
    noisy_img = np.clip(noisy_img1, 0, 1)
    return noisy_img

for i in range(0, 900):
    try:
        #noised images
        img = imread("./normal_normal/train/data_" + str(i) + ".ome.tif")
        imgn = (img - np.min(img)) / (np.max(img) - np.min(img))
        noisyn = add_noise(imgn)
        
        #denoised images
        denoisen = []
        for j in range(32):
            denoisen.append(bm3d.bm3d(noisyn[j], sigma_psd=10/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING))
            print(str(j))
        noisy = noisyn * (np.max(img) - np.min(img)) + np.min(img)
        denoise = denoisen * (np.max(img) - np.min(img)) + np.min(img)
        imwrite("./normal/train/data_" + str(i) + ".ome.tif", denoisen)
        imwrite("./noisy/train/data_" + str(i) + ".ome.tif", noisy)
        print("Yessir! " + str(i))
    except Exception as e:      
        print(e)        

