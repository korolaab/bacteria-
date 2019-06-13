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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

            # cv.drawContours(img, contours, -1, (255,255,0), 1, cv.LINE_AA, hierarchy, 1 )
            # ficv=plt.figure(1)
            # n = n+1
            # plt.imshow(img)
            # print(len(contour/s))
            # print(arr)
            # plt.show()
            # print("123")
            # print(arr)
            # return len(arr)


def d_c(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctlyf
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)
# def iou_loss(true,pred):  #this can be used as a loss if you make it negative
#     true.set_shape(pred.get_shape())
#     intersection = true * pred
#     notTrue = 1 - true
#     union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())
def dice_coef_loss(y_true, y_pred):
        # return -1*K.log(d_c(y_true, y_pred))
        return 1 - d_c(y_true,y_pred)
def val_load(dataset_folder):
    photos = []
    answ = []
    folders = os.listdir(dataset_folder)
    x = 0
    quantity = []
    for folder in folders:
            mask=[]
            # folder = is i
            files = os.listdir(dataset_folder + '/' + folder)
            mask=[]
            for i in files:
                if(i[-9:]=="_mask.jpg"):
                    im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("L")
                    m = np.asarray(im,dtype=np.float32)
                    mask = m/255
                    mask = np.array(mask)
                    mask[mask > 0] = 1
                    answ.append(mask)
                elif(i[-4:]== ".jpg"):

                    im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                    photo = np.asarray(im,dtype=np.float32)/255
                    photos.append(photo)
                # else:
                #     f = open(dataset_folder + '/' + folder + '/'+"q.txt","r")
                #     quantity.append(int(f.read()))
                #     f.close()


    photos = np.array(photos)
    answ = np.array(answ)
    # quantity = np.expand_dims(quantity,axis=1)
    # quantity = np.array(quantity,dtype="float32")
    return photos,answ

def gen(dataset_folder,batch_size):
    folders = os.listdir(dataset_folder)
    n = 0
    while True:
        mask=[]

        folder = random.choice(folders)
        # folder = folders[n]
        files = os.listdir(dataset_folder + '/' + folder)
        # n=1
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
        files = sorted(files)
        # print(files)
        for i in files:
            # print(i[:-9])
            if(i[-9:]=="_mask.jpg"):
                im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("L")
                m = np.asarray(im,dtype=np.float32)
                if(v_f):
                     m = m[::-1, :]
                if(g_f):
                     m = m[:, ::-1]
                m = rotate(m,r,reshape=False)
                # m = m255
                mask = m/255
                mask = np.array(mask)
                mask = np.expand_dims(mask,axis=0)
                mask = np.around(mask)
            elif(i[-4:]==".jpg"):
                im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                im = PIL.ImageEnhance.Brightness(im).enhance(bri)
                im = PIL.ImageEnhance.Contrast(im).enhance(con)
                # im = PIL.ImageEnhance.Color(im).enhance(col)
                # img = np.abs(255-img)
                # im = PIL.ImageEnhance.Sharpness(im).enhance(blur)

                photo = np.asarray(im,dtype="int32")
                # photo = np.array(im,dtype = "int32")
                if(v_f):
                     photo = photo[::-1, :]
                if(g_f):
                     photo = photo[:, ::-1]
                # photo = ndimage.uniform_filter(photo, size=(blur, blur, 1))
                # # photo = np.array(photo,dtype = "float32")
                # photo = np.random.normal(2*photo+2,noise)
                photo = rotate(photo,r,reshape=False)
                # if(negative):
                    # photo = np.abs(255-photo)
                photo = np.array(photo,dtype = np.float32)

                photo = np.expand_dims(photo,axis=0)
            #
            # else:
                # f = open(dataset_folder + '/' + folder + '/'+"q.txt","r")
                # quantity=float32(f.read())

                # f.close


        # print(photo.shape)
        # print(mask.shape)
        # quantity = np.array(quantity,dtype="float32")
        # quantity = np.expand_dims(quantity,axis=1)
        # y = {
        #         "dice": mask,
        #         "count": quantity
        # }
        yield photo,mask

def quantity_loss(true,pred):
    # return tf.math.reduce_mean(pred,axis=0)
    return tf.math.reduce_mean(tf.math.abs(true-pred)/true,axis=0)

# r=np.array([[56],[12]])
# p=np.array([[789],[20]])
# print(K.eval(quantity_loss(K.variable(r), K.variable(p))))
# exit()
def image_func(mask):
            n=0
            mask = np.where(mask > 0.5, 1, 0)
            im1 = np.ascontiguousarray(mask, dtype=np.uint8)
            contours, hierarchy = cv.findContours(im1, cv.RETR_TREE , cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv.contourArea(cnt) > 10:
                    n = n+1
            # return float32(34)
            return float32(n)

def image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = image_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )

    return np.concatenate( results, axis = 0 )
class count_bacteria( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( image_tensor_func,
                           [xin],
                           'float32',
                           stateful=False,
                           name='cvOpt')
        # xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0],1] ) # explicitly set output shape
        return xout
    def compute_output_shape( self, sin ) :
        return ( sin[0], 1 )
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",use_bias=True)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same",use_bias=True)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
def unet_pp_model(pretrained_weights = None,input_shape = (768, 1024,3)):
    n_filters = 32
    batchnorm = False
    dropout = 0.5

    inputs = Input(input_shape,dtype="float32")
    x00 = conv2d_block(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    x00= MaxPooling2D((2, 2)) (x00)
    x00 = Dropout(dropout*0.5)(x00)

    x10 = conv2d_block(x00, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    x10= MaxPooling2D((2, 2)) (x10)
    x10= Dropout(dropout)(x10)

    x20 = conv2d_block(x10, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    x20 = MaxPooling2D((2, 2)) (x20)
    x20 = Dropout(dropout)(x20)

    x30= conv2d_block(x20, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    x30= MaxPooling2D(pool_size=(2, 2)) (x30)
    x30 = Dropout(dropout)(x30)

    x40 = conv2d_block(x30, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    x31 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (x40)
    x31 = concatenate([x31, x30])
    x31= Dropout(dropout)(x31)
    x31 = conv2d_block(x31, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    x22 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (x31)
    x22= concatenate([x22, x20])
    x22= Dropout(dropout)(x22)
    x22 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    x13 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (x22)
    x13 = concatenate([x13, x10])
    x13 = Dropout(dropout)(x13)
    x13 = conv2d_block(x13, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    x04 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    x04= concatenate([x04, x00], axis=3)
    x04= Dropout(dropout)(x04)
    x04= conv2d_block(x04, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    output_mask = Conv2D(1, (1, 1), activation='sigmoid',name = "dice") (c9)
    conv = Conv2D(1,(3,3),activation='linear',padding="same")(output_mask)
    # output_num = count_bacteria(name = "count")(output_mask)
    # model = Model(inputs=[inputs], outputs=output_mask)
    # loss_funcs = {
    #     "dice": dice_coef_loss,
    #     "count":quantity_loss
    #     }
    # lossWeights ={"dice":1,"count":1}
    model = Model(input = inputs, output = output_mask)

    model.compile(optimizer=Adam(lr=1e-6), loss=dice_coef_loss)
    # model.summary()
    # print(model)
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
def unet_model(pretrained_weights = None,input_shape = (768, 1024,3)):
    n_filters = 32
    batchnorm = False
    dropout = 0.5

    inputs = Input(input_shape,dtype="float32")
    c1 = conv2d_block(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

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
    # p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    # p5 = Dropout(dropout)(p5)

    # c6 = conv2d_block(p5, n_filters=n_filters*32, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # ux = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
    # ux = concatenate([ux, c5])
    # ux = Dropout(dropout)(ux)
    # cx = conv2d_block(ux, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)


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

    output_mask = Conv2D(1, (1, 1), activation='linear',name = "dice") (c9)
    # conv = Conv2D(1,(3,3),activation='linear',padding="same")(output_mask)
    # output_num = count_bacteria(name = "count")(output_mask)
    # model = Model(inputs=[inputs], outputs=output_mask)
    # loss_funcs = {
    #     "dice": dice_coef_loss,
    #     "count":quantity_loss
    #     }
    # lossWeights ={"dice":1,"count":1}
    model = Model(input = inputs, output = output_mask)

    model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss)
    model.summary()
    # print(model)
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
unet = unet_model()
# unet = unet_model(pretrained_weights = "/home/korolaab/Documents/bacteria/save/unet_weights-50-0.9977.hdf5")
# print("Saving model to JSON")
json_file = open("models/unet_model.json", "w") ### exporting bc to json
json_string = unet.to_json()
json_file.write(json_string)
json_file.close()
print("Succsess")
exit()
today = datetime.datetime.today()
folder = today.strftime("%Y-%m-%d-%H.%M.%S")
# graph = os.system("mkdir Graph/"+folder)
сheckpoint = ModelCheckpoint('save/unet_weights-{epoch:02d}-{val_loss:.4f}.hdf5',period = 50)
# his = keras.callbacks.TensorBoard(log_dir='Graph/'+today.strftime("%Y-%m-%d")+"/"+ today.strftime("%H.%M.%S"))
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
#                                patience=100, min_lr=1e-5)
x_val,y_val = val_load("val_dataset")
# y_vals = {
        # "dice": y_val,
        # "count": q
# }

# print(x_val.shape)
# # print(y_val.shape)
# unet.fit_generator(gen("dataset",1),
#                     steps_per_epoch = 25,
#                     validation_data=(x_val,y_val),
#                     epochs = 100000,
#                     callbacks=[сheckpoint])

# x_train,y_train = val_load("dataset")

# loss = 0
# for x in range(0,x_train.shape[0]):
#     i =np.expand_dims(x_train[0,:,:],axis=0)
#     l = quantity_loss(y_train[x,:,:],unet.predict(i))
#     loss = loss+ l
#     print(l)
# print("train loss = {}".format(loss/x_train.shape[0]))
#
# print(x_val.shape)
# print(x_val.shape[0])
# loss = 0
# for x in range(0,x_val.shape[0]):
#     i =np.expand_dims(x_val[0,:,:],axis=0)
#     l = quantity_loss(y_val[x,:,:],unet.predict(i))
#     loss = loss+ l
#     print(l)
#     # print(loss)
# print("val loss = {}".format(loss/x_val.shape[0]))
#
# exit()
# from keras.utils import plot_model
# unet.load_weights('save/unet_weights-1850-0.0137.hdf5')
# print("Saving model to PNG")
# plot_model(unet, to_file='unet_model.png',show_shapes=True)



import timeit
# n =0
def finderCV(mask):
            global n
            mask = np.where(mask > 0.99, 1, 0)
            arr = []
            im1 = np.ascontiguousarray(mask, dtype=np.uint8)
            contours, hierarchy = cv.findContours(im1, cv.RETR_TREE , cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                 cord = cv.boundingRect(cnt)
                 arr.append(cord)
            cv.drawContours(img, contours, -1, (255,255,0), 1, cv.LINE_AA, hierarchy, 1 )
            ficv=PLT.figure(1)
            # n = n+1
            PLT.imshow(img)

            # print("123")
            # print(arr)
            return len(contours)
def count_bacteria(img):
    im = np.expand_dims(img,axis=0)

    # fig,ax = PLT.subplots(1)
    x = unet.predict(im)

    # fig1 =PLT.figure(2)
    # PLT.axis('off')
    # PLT.imshow(x[0,:,:,0])


    N=finderCV(x[0,:,:,0])


    return N
import csv
def process_photo(file):
    im = Image.open(file).resize((1024,768))
    img = np.array(im.convert("RGB"))
    green_im = np.copy(img)
    red_im = np.copy(img)
    green_im[:,:,0]=0
    red_im[:,:,1]=0
    green = count_bacteria(green_im)
    red = count_bacteria(red_im)
    return [red,green]
    # start = timeit.default_timer()
#
# outfile = open('count.csv', 'w')
# writer = csv.writer(outfile)
# list_of_files = getListOfFiles("Photo")
# # print(list_of_files[5])
# # exit()
# writer.writerow(["file","red","green"])
# for file in list_of_files:
#     print(file)
#     red,green = process_photo(file)
#
#     writer.writerow([file[6:], red, green])
im = Image.open("/home/korolaab/Documents/bacteria/val_dataset/134/174.jpg")
# im.show()
# im = PIL.ImageEnhance.Color(im).enhance(0)
# im = PIL.ImageEnhance.Brightness(im).enhance(0)
# im = PIL.ImageEnhance.Contrast(im).enhance(0)
img = np.asarray(im.convert("RGB"))
# img = np.abs(255-img)

# # print(os.listdir("dataset_fcn" + '/' + "1"))
# # print(os.listdir("unet_validate" + '/' + "180"))
# blur_kernel = 6
# img = ndimage.uniform_filter(img, size=(blur_kernel, blur_kernel, 1))
# img = skimage.util.random_noise(img, mode='gaussian', seed=12)
# v_min, v_max = np.percentile(img, (0.2, 99.8))
# better_contrast = exposure.rescale_intensity(img, in_range=(v_min, v_max))
# fig1 = PLT.figure(1)
# PLT.imshow(img)
import timeit
im = np.expand_dims(img,axis=0)/255
start = timeit.default_timer()
x = unet.predict(im)
print(x.shape)
fic1=PLT.figure(0)
imgplot=PLT.imshow(x[0,:,:,0])
PLT.colorbar()
imgplot.set_cmap('nipy_spectral')
y=finderCV(x[0,:,:,0])
print(y)
PLT.show()
# L = []
#
#
# for i in y:
#     x,y,w,h = i
#
#     rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor="y",facecolor='none')
#     # PLT.pause(0.001)
#     # Add the patch to the Axes
#     ax.add_patch(rect)
# L = np.array(L)
# L = np.sort(L)
# k = 0
# l=L[k]
#
# print(L)
# if len(L)>0:
#     l_k = L[0]
# else:
#     PLT.show()
#     exit()
# l = l_k
# N_L =[]
# E = 2
# print("Total number: %d"%len(L))
# while (k < len(L)-1):
#     n = 0
#     l = L[k]
#     while(l_k<l+E and k < len(L)-1):
#         n = n +1
#         l_k = L[k]
#         k = k+1
#     N_L.append((n,l))
#
# N_L = np.array(N_L)
# print(N_L)
#
# # print(N_L.shape)
# stop = timeit.default_timer()
# print('Time: ', stop - start)
# graph = PLT.figure(3)
# PLT.plot(N_L[:,1],N_L[:,0],"r")
# PLT.ylabel('Number')
# PLT.xlabel('Length')
#
# PLT.show()
