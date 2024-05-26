import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')

# check if we using GPU
if tf.test.gpu_device_name():

    print("Pracujemy na GPU\n")
    device = '/GPU:0'
else:
    print("Pracujemy na CPU\n")
    device = '/CPU:0'

def autoencoder3D(input_shape, fun):
    
    input_img = Input(shape=input_shape)
    x1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
    x2 = MaxPooling3D((2, 2, 2), padding='same')(x1)
    x3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x2)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x3)

    x4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(encoded)
    x5 = UpSampling3D((2, 2, 2))(x4)
    x6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x5)
    x7 = UpSampling3D((2, 2, 2))(x6)
    decoded = Conv3D(1, (3, 3, 3), activation=fun, padding='same')(x7)

    
    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')  
    return model

input_shape = (32, 512, 512, 1)
data_dir_y = "/home/szymczakmchtr/normal/train"
data_dir_x = "/home/szymczakmchtr/noisy/train"
val_dir_y = "/home/szymczakmchtr/normal/val"
val_dir_x = "/home/szymczakmchtr/noisy/val"
batch_size = 1
shuffle_data = True

data_generator = DataGenerator(data_dir_x, data_dir_y, batch_size=batch_size, shuffle_data=shuffle_data)
val_generator = DataGenerator(val_dir_x, val_dir_y, batch_size=batch_size, shuffle_data=shuffle_data)
epochs = 10
model = autoencoder3D(input_shape, 'relu')
model.fit(data_generator, validation_data=val_generator, epochs=epochs)
model.save("autoencoder3D_e10_relu_32_64.h5")

