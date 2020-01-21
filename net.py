#!/usr/bin/env python3
import gzip
import hashlib
import numpy as np
import os
import keras
from keras.models import load_model , model_from_json
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import datetime
import io
import imageio
import cv2
import socket
import keras.backend as K
dtype = "float32"
K.set_floatx(dtype)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def load_model():
    json_file = open("models/model2.json", "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("models/model3_weights.hdf5")       ###loading weights from file
    return model

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

def recognize(image):
    shift = 200
    size = 200
    output = np.zeros([image.shape[0],image.shape[1]],dtype="float32")
    buf1 = np.zeros([1,size,size,3],dtype="float32")
    buf2 = np.zeros([size,size],dtype="float32")
    buf3=np.zeros([size,size],dtype="float32")
    for x_shift in range(math.ceil(image.shape[0]/shift)):
        for y_shift in range(math.ceil(image.shape[1]/shift)):
            x = shift*x_shift
            y = shift*y_shift
            piece = image[x:x+size,y:y+size,:]
            buf1[0,:piece.shape[0],:piece.shape[1],:]=piece
            buf2=(model.predict(buf1))[0,:,:,0]
            output[x:x+piece.shape[0],y:y+piece.shape[1]]+=buf2[:piece.shape[0],:piece.shape[1]]
    return output
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
            img_bmp_raw=gzip.decompress(raw_img) #decompress image
            filebytes = np.asarray(bytearray(img_bmp_raw),dtype=np.uint8)
            img = cv2.imdecode(filebytes,cv2.IMREAD_UNCHANGED)            
            start = time.time()
            y = recognize(img)
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
            print("sending prediction to:".format(addr))
            conn.send(len_comp_img+comp_img+len_check_sum+check_sum)
        #send prediction to webserver
        conn.close()
