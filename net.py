#!/usr/bin/env python3
import gzip
import struct
import hashlib
import numpy as np
import os
import math
import keras
from keras.models import load_model , model_from_json
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import PIL
from PIL import Image
import datetime
import io
import imageio
import cv2
import csv
import socket
import keras.backend as K
dtype = "float16"
K.set_floatx(dtype)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def crop_from_photo(image,size,shift,mode):
     N = math.ceil(image.shape[0]/shift)*math.ceil(image.shape[1]/shift)
     if mode == 3:
         pieces = np.zeros([N,size,size,3],dtype=dtype)
     else:
         pieces = np.zeros([N,size,size,1],dtype=dtype)
         image = image[:,:,2]
         image = np.expand_dims(image,axis=2)
     n=0
     for x_shift in range(math.ceil(image.shape[0]/shift)):
         for y_shift in range(math.ceil(image.shape[1]/shift)):
             x = shift*x_shift
             y = shift*y_shift
             piece = image[x:x+size,y:y+size,:]
             pieces[n,:piece.shape[0],:piece.shape[1],:]=piece
             n+=1
     return pieces

def photo_from_crop(arr,shift,x_size,y_size):
    arr = (arr*100).astype(np.int16)
    image = np.zeros([x_size,y_size],dtype=np.int16)
    n = 0
    size = arr.shape[1]

    for x_shift in range(math.ceil(x_size/shift)):
        x = shift*x_shift
        for y_shift in range(math.ceil(y_size/shift)):
            y = shift*y_shift
            piece = image[x:x+size,y:y+size]
            image[x:x+piece.shape[0],y:y+piece.shape[1]]=arr[n,:piece.shape[0],:piece.shape[1],0]

            piece = image[x:x+shift,y:y+shift]
            ins_x = piece.shape[0]
            ins_y = piece.shape[1]
            crop = arr[n,:ins_x,:ins_y,0]
            image[x:x+ins_x,y:y+ins_y] = (np.bitwise_and(piece,crop) + np.bitwise_or(piece,crop))/2
            n+=1
    return image

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

def load_model():
    json_file = open("models/model2.json", "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("models/model2_weights.hdf5")       ###loading weights from file
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
        return 0
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
def recvall(sock,n):
    data=bytearray()
    while len(data)<n:
        packet=sock.recv(n-len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
def recv_msg(sock):
    raw_msglen=recvall(sock,8)
    if not raw_msglen:
        return None
    msglen=struct.unpack(">Q",raw_msglen)[0]
    return  recvall(sock,msglen)

def recognize(img_bytes):
    global model
    shift = 180
    size = 200
    #crop image
    img = Image.open(img_bytes)
    im = np.array(img.convert("RGB"),dtype="float32")
    im = im[:,:,[2,1,0]]
    crop = crop_from_photo(im,size,shift,3)
    #predict
    print(crop.shape)
    start = time.time()
    output = model.predict(crop)
    end = time.time()
    print("net takes:%f"%(end-start))
    #compile image from crop
    predict = photo_from_crop(output,shift,im.shape[0],im.shape[1])
    return predict
import time
if __name__ == "__main__":
    model = load_model()
    model.summary()
    print("model ready")
    sock=socket.socket()
    sock.bind(('127.0.0.1',9992))
    sock.listen(1)
    while True:
        conn,addr=sock.accept()
        raw_img=recv_msg(conn)
        check_sum=recv_msg(conn)
        print("client: {}".format(addr))
        #get bytes data from webserver(8bytes+bytes_image+8bytes+hash)
        #check_sum compare
        h=hashlib.md5(raw_img)
        if(check_sum == h.digest()): 
            img_bmp=gzip.decompress(raw_img)
            #decompress image
            start = time.time()
            y = recognize(io.BytesIO(img_bmp))
            end = time.time()
            print("pred takes: %f s"%(end-start))
            #net predict
            img_buf=io.BytesIO()
            imageio.imwrite(uri=img_buf,im=y,format="PNG")
            img_bmp=img_buf.getvalue()
            comp_img=gzip.compress(img_bmp)
            len_comp_img=struct.pack(">Q",len(comp_img))
            #int len to bytes
            h=hashlib.md5(comp_img)
            check_sum = h.digest()
            len_check_sum= struct.pack(">Q",len(check_sum))
                # print(len(len_comp_img))
                # print(len(comp_img))
                # print(len(len_check_sum))
                # print(len(check_sum))
            print("sending prediction to:".format(addr))
            conn.send(len_comp_img+comp_img+len_check_sum+check_sum)
    #send prediction to webserver
        conn.close()
