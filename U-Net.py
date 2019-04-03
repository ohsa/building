# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:40:32 2018

@author: wusiao
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np  
from keras.models import Sequential  
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.layers.merge import concatenate
from PIL import Image  
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

import cv2
import random
import os
from tqdm import tqdm  
import tensorflow as tf
from keras.optimizers import SGD

import keras.backend.tensorflow_backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


img_w = 256  
img_h = 256  
 
n_label = 2

  
classes = [0. , 255.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  



        
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img


filepath ='/data160/'  

def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set

# data for training  
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img) 
            train_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
# data for validation 
def generateValidData(batch_size,data=[]):  
    #print 'generateValidData...'
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1
#            img = load_img(filepath + 'src/' + url, target_size=(img_w, img_h))
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  
                
def unet():
    
    inputs = Input(( img_w, img_h,3))      

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    print ('conv1:',conv1.shape)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    print ('conv1:',conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv1)
    print ('pool1:',pool1.shape)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    print ('conv2:',conv2.shape)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    print ('conv2:',conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv2)
    print ('pool2:',pool2.shape)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    print ('conv32:',conv3.shape)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    print ('conv3:',conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv3)
    print ('pool3:',pool3.shape)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    print ('conv4:',conv4.shape)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    print ('conv4:',conv4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv4)
    print ('pool4:',pool4.shape)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    print ('conv5:',conv5.shape)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)
    print ('conv5:',conv5.shape)

    up6 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv5), conv4], axis=3)   
    print ('up6:',up6.shape)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    print ('conv6:',conv6.shape)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    print ('conv6:',conv6.shape)

    up7 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv6), conv3], axis=3)
    print ('up7:',up7.shape)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    print ('conv7:',conv7.shape)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    print ('conv7:',conv7.shape)

    up8 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv7), conv2], axis=3)
    print ('up8:',up8.shape)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    print ('conv8:',conv8.shape)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    print ('conv8:',conv8.shape)

    up9 = concatenate([UpSampling2D(size=(2, 2),data_format='channels_last')(conv8), conv1], axis=3)
    print ('up9:',up9.shape)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    print ('conv9:',conv9.shape)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    print ('conv9:',conv9.shape)

    conv10 = Conv2D(2, (1, 1), activation="sigmoid")(conv9)
    
    print ('conv10:',conv10.shape)

 

    r11=Reshape((img_w*img_h,2))(conv10)
    print ('r11:',r11.shape)
    model = Model(inputs=inputs, outputs=r11)

    sgd = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


  
def train(args): 
    EPOCHS = 30
    BS = 16
    model = unet()  
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')  
    callable = [modelcheck]  
    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable,max_q_size=1)  

    # plot the training loss and accuracy
    myfont =  FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',size=10)
    rcParams['axes.unicode_minus']=False 
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.yticks(np.linspace(0,1,21,endpoint=True))  # 设置y轴刻度

    plt.plot(np.arange(0, N), H.history["loss"],color="blue", label=u"训练损失")
    plt.plot(np.arange(0, N), H.history["val_loss"],color="red", label=u"验证损失")
    plt.plot(np.arange(0, N), H.history["acc"],color="green", label=u"训练精确度")
    plt.plot(np.arange(0, N), H.history["val_acc"], color="black",label=u"验证精确度")
#    plt.title("U-Net")
    plt.xlabel(u"迭代次数",fontproperties=myfont)
    plt.ylabel(u"损失/精确度",fontproperties=myfont)
    plt.legend(loc=3,bbox_to_anchor=(0., 1.02, 1., .102),ncol=4,prop =myfont,mode='expand', borderaxespad=0.)
    plt.savefig(args["plot"])

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augment", help="using data augment or not",
                    action="store_true", default=False)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    if args['augment'] == True:
        filepath ='data160/'

    train(args)  
    #predict() 
