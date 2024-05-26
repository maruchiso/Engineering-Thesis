def autoencoder3D(input_shape, fun):

    input_img = Input(shape=input_shape)
    x0 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
    #x0 = BatchNormalization()(x0)
    x1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x0)
    #x1 = BatchNormalization()(x1)
    x2 = MaxPooling3D((2, 2, 2), padding='same')(x1)
    x3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x2)
    #x3 = BatchNormalization()(x3)
    encoded = MaxPooling3D((2, 2, 2), padding='same')(x3)
    #drop1 = Dropout(0.2)(encoded)

    x4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(encoded)
    #x4 = BatchNormalization()(x4)
    x5 = UpSampling3D((2, 2, 2))(x4)
    x6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x5)
    #x6 = BatchNormalization()(x6)
    x7 = UpSampling3D((2, 2, 2))(x6)
    x8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x7)
    #x8 = BatchNormalization()(x8)
    x9 = UpSampling3D((2, 2, 2))(x8)
    #drop2 = Dropout(0.2)(x9)

    decoded = Conv3D(1, (3, 3, 3), activation=fun, padding='same')(x9)

    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')  
    return model

