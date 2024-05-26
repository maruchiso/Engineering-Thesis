def unet_dropout(input_shape, fun):
    
    input_img = Input(shape=input_shape)
    x1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    x3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x2)
    x4 = MaxPooling3D(pool_size=(2, 2, 2))(x3)
    
    drop1 = Dropout(0.2)(x4)
    
    x5 = Conv3D(32, (2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(drop1))
    x6 = concatenate([x3, x5], axis=-1)
    x7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x6)
    x8 = Conv3D(16, (2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(x7))
    x9 = concatenate([x1, x8], axis=-1)
    
    drop2 = Dropout(0.2)(x9)
    
    decoded = Conv3D(1, (1, 1, 1), activation=fun)(drop2)
    

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

