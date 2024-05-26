import numpy as np                                            
from keras.models import load_model
from tifffile import imread, imwrite

def denormalize_3d_image(normalized_img, original_min, original_max):
    denormalized_img = normalized_img * (original_max - original_min) + original_min
    return denormalized_img

def normalize_3d_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

#model's name
x = "unet3D_e10_relu_32_64"
model = load_model("./" + x + ".h5")

#prepare date for prediction
single_file_path_x = "./noisy/test/data_597.ome.tif"  
single_file_x = imread(single_file_path_x)
single_file_x_normalized = normalize_3d_image(single_file_x)  
single_file_x_normalized = np.expand_dims(single_file_x_normalized, axis=0)

#prediction
predicted_single_file_y = model.predict(single_file_x_normalized)

#come back to original shape
predicted_single_file_y1 = np.squeeze(predicted_single_file_y, axis=0)
predicted_single_file_y2 = np.squeeze(predicted_single_file_y1, axis=-1)
predicted_single_file_y3 = denormalize_3d_image(predicted_single_file_y2, np.min(single_file_x), np.max(single_file_x))
imwrite("./path/to/save/predicted/file.ome.tif", predicted_single_file_y3)
print(predicted_single_file_y3.shape)
print(np.max(predicted_single_file_y3), np.min(predicted_single_file_y3))

