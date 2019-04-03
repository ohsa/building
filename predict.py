# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:45:41 2018

@author: wusiao
"""


import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['austin33.png','austin34.png','austin35.png','austin36.png','chicago20.png','chicago27.png','chicago28.png','chicago35.png','kitsap24.png','kitsap25.png','kitsap31.png','kitsap32.png','tyrol-w33.png','tyrol-w34.png','tyrol-w35.png','tyrol-w36.png','vienna33.png','vienna34.png','vienna35.png','vienna36.png']

image_size = 256

classes = [0. ,  255.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

#def predict_classes(self, x, batch_size=32, verbose=1):
def predict_classes(pred):
    '''Generate class predictions for the input samples
    batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions.
    '''
#    proba = self.predict(x, batch_size=batch_size, verbose=verbose)
    if pred.shape[-1] > 1:
        return pred.argmax(axis=-1)
    else:
        return (pred > 0.5).astype('int32')
    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread('E:/test20/' + path)
        # pre-process the image for classification
        #image = image.astype("float") / 255.0
        #image = img_to_array(image)
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print ('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
#                crop = padding_img[:3,i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]
                print ('crop:',crop.shape)
                ch,cw,_ = crop.shape
#                if ch != 256 or cw != 256:
#                    print ('invalid size!')
#                    continue
                    
                crop = np.expand_dims(crop, axis=0)
                print ('crop:',crop.shape)
                pred = model.predict(crop,verbose=2)   
                pred = predict_classes(pred)
                print ('pred:',pred)
#                pred[pred > 0.5] = 1
#                pred[pred <= 0.5] = 0
                pred = labelencoder.inverse_transform(pred[0])  
#                print (np.unique(pred))  
                pred = pred.reshape((256,256)).astype(np.uint8)
                print ('pred:',pred.shape)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]
#                mask_whole = mask_whole.astype("float") / 255.0
##                mask_whole = mask_whole.astype("float")
#                print ('mask_whole:',mask_whole.shape)
                

        
        
        
        cv2.imwrite('E:/test20_predict/austin34'+str(n+1)+'.png',mask_whole[0:h,0:w])
        
    

    
if __name__ == '__main__':
    args = args_parse()
    predict(args)
    
    



