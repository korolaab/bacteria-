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
import cv2 as cv
import csv
import imageio
import socket
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    global model
    model = model_from_json(model_json)
    if not(weights is None):
     model.load_weights(weights)       ###loading weights from file
     
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
def predict(im):
    img = np.asarray(im.convert("RGB").resize((1024,768)))
    im = np.expand_dims(img,axis=0)/255
    x = model.predict(im)
    x = x[0,:,:,:]
    x = x.reshape((768,1024))
    return  Image.fromarray(x,mode="L")
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
if __name__ == "__main__":
    #graph = tf.get_default_graph()
    #sess =tf.Session()
    #set_session(sess)
    load_model()
    print("model ready")
    sock=socket.socket()
    sock.bind(('127.0.0.1',9992))
    sock.listen(1)
    while True:
        conn,client_address=sock.accept()
        raw_img=recv_msg(conn)
        print(len(raw_img))
        conn.send(b"")
        check_sum=recv_msg(conn)
        #get bytes data from webserver
        #check_sum compare
        h=hashlib.md5(raw_img)
        if(check_sum == h.digest()): 
            img_bmp=gzip.decompress(raw_img)
            im = Image.open(io.BytesIO(img_bmp))               
            #convert from base64 to pillow object
            prediction = predict(im)        
            #net predict
            img_buf=io.BytesIO()
            prediction.save(img_buf,"BMP")
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
            conn.send(len_comp_img+comp_img+len_check_sum+check_sum)

	#send prediction to webserver
        conn.close()
