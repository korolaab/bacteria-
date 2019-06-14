import numpy as np
import os
import sys
import math
import keras
from keras.models import load_model , model_from_json
import PIL
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFilter
import datetime

from matplotlib import pyplot as PLT
import argparse
import cv2 as cv
import csv
import timeit
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def finderCV(mask,img):
            global n
            mask = np.where(mask > 0.99, 1, 0)
            arr = []
            im1 = np.ascontiguousarray(mask, dtype=np.uint8)
            contours, hierarchy = cv.findContours(im1, cv.RETR_TREE , cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if(cv.contourArea(cnt)>10):
                    cord = cv.boundingRect(cnt)
                    arr.append(cord)
            cv.drawContours(img, contours, -1, (255,255,0), 1, cv.LINE_AA, hierarchy, 1 )
            ficv=PLT.figure(1)

            PLT.imshow(img)

            return arr

def count_bacteria(img):
    im = np.expand_dims(img,axis=0)
    x = unet.predict(im)
    N=finderCV(x[0,:,:,0])
    return N

def load_model(weights=None):
    json_file = open("models/unet_model.json", "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    if(D):
        model.summary()
    if(weights is None):
        return model
    model.load_weights(weights)       ###loading weights from file
    return model

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


def save_csv(data):
    np.savetxt(args.file_name, data, delimiter=",",header="quantity,length",fmt='%d,%.2f')
def main(file):
    im = Image.open(file)
    img = np.asarray(im.convert("RGB"))
    im = np.expand_dims(img,axis=0)/255
    start = timeit.default_timer()
    x = unet.predict(im)
    fic1=PLT.figure(0)
    imgplot=PLT.imshow(x[0,:,:,0])
    PLT.colorbar()
    imgplot.set_cmap('nipy_spectral')
    y=finderCV(x[0,:,:,0],img)
    print(len(y))
    if(args.length):
        dist = distribiution_length(y)
        show_plot(dist)
    if(args.file_name):
        print("save data to {}".format(args.file_name))
        save_csv(dist)


def draw_rect():

        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor='none')
        # PLT.pause(0.001)
        # Add the patch to the Axes
        ax.add_patch(rect)

def distribiution_length(y): # the length distribution of the number of bacteria
    L=[]
    for i in y:
        x,y,w,h = i

        L.append((w**2+h**2)**0.5)

    L = np.array(L)
    L = np.sort(L)
    k = 0
    l=L[k]

    # print(L)
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
    return N_L

def show_plot(N_L):
    # print(N_L.shape)
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    graph = PLT.figure(3)
    PLT.plot(N_L[:,1],N_L[:,0],"r")
    PLT.ylabel('Number')
    PLT.xlabel('Length')








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding objects on image")
    parser.add_argument("--image",action = "store", metavar='<path>',default = None, required=True ,dest = "img", help="Image")
    # parser.add_argument("--model",dest='model',default = "cnn", choices=["cnn","unet"], help="Models")
    parser.add_argument("--weights",action = "store", metavar='<path>', dest = "w", help="Weights")
    parser.add_argument("--Debug ",dest='Debug', action="store_true", help="Debuging information and models parameters")
    parser.add_argument("--dist_length",dest="length",action="store_true",help="Show the length distribution of the number of bacteria")
    parser.add_argument("--save_csv",dest="file_name",action="store",metavar="<path>",default="None",help="Save data to csv file")
    # parser.add_argument()
    #to csv
    args = parser.parse_args()
    # m = args.model
    D = args.Debug
    w = args.w
    if(not D):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no TF debugging info


    if(w == None):
        print(bcolors.WARNING + "No pretrained weights the results can be unpredictable"+ bcolors.ENDC)
    unet = load_model(weights = w)
    main(args.img)
    PLT.show()
