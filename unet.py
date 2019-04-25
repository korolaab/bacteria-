
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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
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
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def d_c(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctlyf
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)

def dice_coef_loss(y_true, y_pred):
        # return -1*K.log(d_c(y_true, y_pred))
        return 1 - d_c(y_true,y_pred)
def val_load(dataset_folder):
    photos = []
    answ = []
    folders = os.listdir(dataset_folder)
    x = 0
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
                else:
                    im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                    photo = np.asarray(im,dtype=np.float32)/255

            photos.append(photo)
            answ.append(mask)
    photos = np.array(photos)
    answ = np.array(answ)

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
        bri = random.uniform(0.5, 1)
        con = random.uniform(0.5, 1)
        col = random.uniform(0, 1)

        blur = random.uniform(0, 1)
        # noise = random.randint(1, 60)

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
            else:
                im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                im = PIL.ImageEnhance.Brightness(im).enhance(bri)
                im = PIL.ImageEnhance.Contrast(im).enhance(con)
                im = PIL.ImageEnhance.Color(im).enhance(col)
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
                photo = np.array(photo,dtype = np.float32)/255
                photo = np.expand_dims(photo,axis=0)

        mask = np.array(mask)
        mask = np.expand_dims(mask,axis=0)
        mask = np.around(mask)
        # print(photo.shape)
        # print(mask.shape)
        yield photo,mask


# folder = random.choice(os.listdir("dataset"))
#
# files = os.listdir("dataset_fcn" + '/' + folder)
# name=[]
# for i in files:
#     if i[:-4] =="photo":
#         continue
#     else:
#         name.append(i[:-4])
# name = sorted(name)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_model(pretrained_weights = None,input_shape = (768, 1024,3)):
    n_filters = 32
    batchnorm = False
    dropout = 0.4

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

    # expansive path
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

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    # model = Model(inputs=[inputs], outputs=[outputs])

    model = Model(input = inputs, output = outputs)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[d_c])
    # model.summary()
    # print(model)
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
# unet = unet_model()
unet = unet_model(pretrained_weights = "/home/korolaab/Documents/bacteria/save/unet_weights-200-0.5580.hdf5")
print("Saving model to JSON")
json_file = open("unet_model.json", "w") ### exporting bc to json
json_string = unet.to_json()
json_file.write(json_string)
json_file.close()
print("Succsess")
graph = os.listdir("Graph")
for i in graph:
    os.remove("Graph/"+i)
# r_l=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                                       patience=100, verbose=0, mode='auto',
#                                       min_delta=0.0001, cooldown=0, min_lr=0)
# earlystopper=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta= 0.0001,mode='auto' ,patience=1000,verbose=1)
# сheckpoint = ModelCheckpoint('save/unet_weights-{epoch:02d}-{val_loss:.4f}.hdf5',period = 50)
# his = keras.callbacks.TensorBoard(log_dir='./Graph')
# x_val,y_val = val_load("val_dataset")
# print(x_val.shape)
# print(y_val.shape)
# unet.fit_generator(gen("dataset",1),
#                     steps_per_epoch = 25,
#                     validation_data=(x_val,y_val),
#                     epochs = 10000,
#                     callbacks=[сheckpoint,his])
# from keras.utils import plot_model
# unet.load_weights('save/unet_weights-1850-0.0137.hdf5')
# print("Saving model to PNG")
# plot_model(unet, to_file='unet_model.png',show_shapes=True)

im = Image.open("/home/korolaab/Documents/bacteria/val_dataset/133/150.jpg")
# im.show()
img = np.asarray(im.convert("RGB"))
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
# x = np.around(x)
fig,ax = PLT.subplots(1)
ax.imshow(img)


def finder(img):

    arr =[]
    # print(img.shape)
    contours = measure.find_contours(img, 0.9)
    for contour in contours:
        coordinates = contour.astype(int)
        ymax, xmax = coordinates.max(axis=0)
        ymin, xmin = coordinates.min(axis=0)
        # print("xmin{} ymin{} xmax{} ymax{}".format(xmin,ymin,xmax,ymax))
        cord = (xmin,ymin,xmax-xmin,ymax-ymin)
        arr.append(cord)
    return arr
fig1 =PLT.figure(2)
PLT.axis('off')
PLT.imshow(x[0,:,:,0])


y=finder(x[0,:,:,0])
L = []


for i in y:
    x,y,w,h = i

    if(w*h<100): #square checking
        continue
    if(w/h>10 or h/w>10): #difference in width and length
        continue
    c = np.mean(img[x:x+w,y:y+h], axis=(0, 1))
    if(c[1]>c[0]):
        color = "g"
    else:
        color = "r"
    L.append((w**2+h**2)**0.5)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor='none')
    # PLT.pause(0.001)
    # Add the patch to the Axes
    ax.add_patch(rect)
L = np.array(L)
L = np.sort(L)
k = 0
l=L[k]

print(L)
if len(L)>0:
    l_k = L[0]
else:
    PLT.show()
    exit()
l = l_k
N_L =[]
E = 2
print("Total number: %d"%len(L))
while (k < len(L)-1):
    n = 0
    l = L[k]
    while(l_k<l+E and k < len(L)-1):
        n = n +1
        l_k = L[k]
        k = k+1
    N_L.append((n,l))

N_L = np.array(N_L)
print(N_L)

# print(N_L.shape)
stop = timeit.default_timer()
print('Time: ', stop - start)
graph = PLT.figure(3)
PLT.plot(N_L[:,1],N_L[:,0],"r")
PLT.ylabel('Number')
PLT.xlabel('Length')

PLT.show()
# while True:
#     img, x = next(gen("dataset",1))
#     img = np.array(img,dtype="int32")
#     # fig1= PLT.figure(1)
#
#     # img = np.expand_dims(img,axis=0)/255
#     # x = unet.predict(img)
#     # x = np.around(x)
#
#
#     fig = PLT.figure(2)
#     # x = np.where(x > 0, 1, 0)
#     # print(x.shape)
#
#     PLT.subplot(1,2,1)
#     # print(x[0,:,:,i].shape)
#     PLT.axis('off')
#     PLT.imshow(x)
#     # PLT.colorbar()
#     PLT.subplot(1,2,2).set_title("photo")
#     PLT.imshow(img)
#     PLT.show()
