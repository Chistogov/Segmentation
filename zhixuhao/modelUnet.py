import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)

def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    #
    # up6 = Conv2D(512, 2, activation='relu', padding='same')(
    #     UpSampling2D(size=(2, 2))(drop5))
    # merge6 = concatenate([drop4, up6])
    # conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    #
    # up7 = Conv2D(256, 2, activation='relu', padding='same')(
    #     UpSampling2D(size=(2, 2))(conv6))
    # merge7 = concatenate([conv3, up7])
    # conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    #
    # up8 = Conv2D(128, 2, activation='relu', padding='same')(
    #     UpSampling2D(size=(2, 2))(conv7))
    # merge8 = concatenate([conv2, up8])
    # conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    #
    # up9 = Conv2D(64, 2, activation='relu', padding='same')(
    #     UpSampling2D(size=(2, 2))(conv8))
    # merge9 = concatenate([conv1, up9])
    # conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    concat_axis = 3

    conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    # ch, cw = get_crop_shape(inputs, conv9)
    # conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
    # conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)

    # flatten = Flatten()(conv9)
    # Dense1 = Dense(512, activation='relu')(flatten)
    # BN = BatchNormalization()(Dense1)
    # Dense2 = Dense(17, activation='sigmoid')(BN)

    model = Model(input=inputs, output=conv10)
    learning_rate = 0.001
    optim = Adam(lr=learning_rate)
    # optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# UserWarning: Possible precision loss when converting from float32 to uint16
