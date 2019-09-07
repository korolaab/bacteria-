import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys

from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler , ReduceLROnPlateau
from keras import backend as keras
import math
from keras.models import *

import keras.backend as K
import tensorflow as tf

import keras

import PIL
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFilter
import datetime

from keras.models import load_model , model_from_json
import tensorflow as tf
import argparse
import numpy as np
# import load_photo as LOAD
import os
from keras.utils import plot_model
from matplotlib import pyplot as PLT
from matplotlib import pyplot, transforms
import scipy.misc
from scipy.ndimage import rotate
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import random
import skimage
from skimage import measure
import matplotlib.patches as patches
import cv2 as cv

def d_c(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctlyf
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    # return K.mean(intersection / union)



    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
def dice_coef_loss(y_true, y_pred):
        # return -1*K.log(d_c(y_true, y_pred))
        return 1 - d_c(y_true,y_pred)
def val_load(dataset_folder):
    folders = os.listdir(dataset_folder)
    n = 0
    files_masks = os.listdir(dataset_folder+"/masks")
    # files_images=os.listdir(dataset_folder+"/images")
    photo_array=[]
    mask_array = []
    for i in range(1,len(files_masks)):


        im = (Image.open(dataset_folder+'/masks/{}.jpg'.format(i))).convert("L")
        m = np.asarray(im,dtype=np.float32)
        mask = m/255
        mask = np.array(mask)
        mask = np.around(mask)
        mask_array.append(mask)

        im = (Image.open(dataset_folder + '/images/{}.jpg'.format(i))).convert("RGB")
        photo = np.array(im,dtype = np.float32)
        photo_array.append(photo)

    photo_array = np.array(photo_array)
    mask_array = np.array(mask_array)
    return photo_array,mask_array

def gen(dataset_folder,batch_size):
    folders = os.listdir(dataset_folder)
    n = 0
    while True:
        mask=[]
        # folder = folders[n]
        files_masks = os.listdir(dataset_folder+"/masks")
        files_images=os.listdir(dataset_folder+"/images")
        num = random.randint(1, len(files_masks))
        v_f = bool(random.getrandbits(1))
        g_f = bool(random.getrandbits(1))
        bri = random.uniform(0.5, 1.2)
        con = random.uniform(0.5, 1.2)
        col = random.uniform(0, 1.2)
        quantity = 0
        blur = random.uniform(0, 1)
        # noise = random.randint(1, 60)
        negative = bool(random.getrandbits(1))
        r = random.uniform(-40,40)
        # files = sorted(files)
        # print(files)

        im = (Image.open(dataset_folder+'/masks/{}.jpg'.format(num))).convert("L")
        m = np.asarray(im,dtype=np.float32)
        if(v_f):
             m = m[::-1, :]
        if(g_f):
             m = m[:, ::-1]
        m = rotate(m,r,reshape=False)

        mask = m/255
        mask = np.array(mask)
        mask = np.expand_dims(mask,axis=0)
        mask = np.around(mask)

        im = (Image.open(dataset_folder + '/masks/{}.jpg'.format(num))).convert("RGB")
        im = PIL.ImageEnhance.Brightness(im).enhance(bri)
        im = PIL.ImageEnhance.Contrast(im).enhance(con)
        # im = PIL.ImageEnhance.Color(im).enhance(col)
        # img = np.abs(255-img)
        # im = PIL.ImageEnhance.Sharpness(im).enhance(blur)

        photo = np.asarray(im,dtype="int32")

        if(v_f):
             photo = photo[::-1, :]
        if(g_f):
             photo = photo[:, ::-1]

        photo = rotate(photo,r,reshape=False)
        photo = np.array(photo,dtype = np.float32)
        photo = np.expand_dims(photo,axis=0)

        yield photo,mask



def quantity_loss(true,pred):
    # return tf.math.reduce_mean(pred,axis=0)
    return tf.math.reduce_mean(tf.math.abs(true-pred)/true,axis=0)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",use_bias=True)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x =keras.layers.LeakyReLU(alpha=0)(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",use_bias=True)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x =keras.layers.LeakyReLU(alpha=0)(x)
    return x

def unet_model(pretrained_weights = None,input_shape = (768, 1024,3)):
    n_filters = 32
    batchnorm = True
    dropout = 0.4

    inputs = Input(input_shape,dtype="float32")
    c1 = conv2d_block(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    mask = Conv2D(1, (1, 1),name = "dice") (c9)
    output_mask = Activation("sigmoid")(mask)
    model = Model(input = inputs, output = output_mask)

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss)
    model.summary()
    # print(model)
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def train(Weights,name_json_file,Tensor_board_logs=False):
    unet = unet_model(Weights)
    json_file = open(name_json_file, "w")
    json_string = unet.to_json()
    json_file.write(json_string)
    json_file.close()
    сheckpoint = ModelCheckpoint('save/unet_weights-{epoch:02d}-{val_loss:.4f}.hdf5',period = 50)
    if(Tensor_board_logs):
            today = datetime.datetime.today()
            folder = today.strftime("%Y-%m-%d-%H.%M.%S")
            his = keras.callbacks.TensorBoard(log_dir='Graph/'+today.strftime("%Y-%m-%d")+"/"+ today.strftime("%H.%M.%S"))
            callbacks = [сheckpoint,his]
    else:
        callbacks=[сheckpoint]
    x_val,y_val = val_load("Dataset/validate")
    print(x_val.shape)

    unet.fit_generator(gen("Dataset/train",1),
                        steps_per_epoch = 25,
                        validation_data=(x_val,y_val),
                        epochs = 10000,
                        callbacks=callbacks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument("--json_file_model",action = "store",default="model.json", metavar='<path>', dest = "json", help="Save model to json file")
    parser.add_argument("--weights",action = "store", metavar='<path>', dest = "w", help="Weights")
    parser.add_argument("--tb",action = "store_true", dest = "tb", help="TensorBoard logging")
    parser.add_argument("-d", "--Debug ", dest='Debug', action="store_true", help="Debuging information and models parameters")
    args = parser.parse_args()
    D = args.Debug

    if(not D):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no TF debugging info



    train(args.w,args.json,args.tb)
