import os
import numpy as np
from keras.utils import Sequence
from sklearn.utils import shuffle
from tifffile import imread, imwrite
from glob import glob

class DataGenerator(Sequence):
    def __init__(self, data_dir_x, data_dir_y, batch_size=32, shuffle_data=True):
        self.data_dir_x = data_dir_x
        self.data_dir_y = data_dir_y
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.image_filenames = glob(os.path.join(data_dir_x, '*.ome.tif'))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.image_filenames) // self.batch_size
    
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_x, batch_y = self.load_data(start_idx, end_idx)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle_data:
            self.image_filenames = shuffle(self.image_filenames)

    def load_data(self, start_idx, end_idx):
        batch_x = []
        batch_y = []

        for filename in self.image_filenames[start_idx:end_idx]:
            zaszumione_path = os.path.join(self.data_dir_x, filename)
            odszumione_path = os.path.join(self.data_dir_y, filename)

            if not (os.path.exists(zaszumione_path) and os.path.exists(odszumione_path)):
                continue

            zaszumione_img1 = imread(zaszumione_path)
            zaszumione_img = self.normalize_3d_image(zaszumione_img1)

            odszumione_img1 = imread(odszumione_path)
            odszumione_img = self.normalize_3d_image(odszumione_img1)

            batch_x.append(np.expand_dims(zaszumione_img, axis=-1))
            batch_y.append(np.expand_dims(odszumione_img, axis=-1))
            print(np.array(batch_x).shape)

        return np.array(batch_x), np.array(batch_y)

    def normalize_3d_image(self, img):
        img_min = np.min(img)
        img_max = np.max(img)
        normalized_img = (img - img_min) / (img_max - img_min)
        return normalized_img
    
