# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:48:43 2018

@author: wusiao
"""

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 256  
img_h = 256  

image_sets = ['001.png','002.png','003.png','004.png','005.png','006.png','007.png','008.png','009.png','010.png',
              '011.png','012.png','013.png','014.png','015.png','016.png','017.png','018.png','019.png','020.png',
              '021.png','022.png','023.png','024.png','025.png','026.png','027.png','028.png','029.png','030.png',
              '031.png','032.png','033.png','034.png','035.png','036.png','037.png','038.png','039.png','040.png',
              '041.png','042.png','043.png','044.png','045.png','046.png','047.png','048.png','049.png','050.png',
              '051.png','052.png','053.png','054.png','055.png','056.png','057.png','058.png','059.png','060.png',
              '061.png','062.png','063.png','064.png','065.png','066.png','067.png','068.png','069.png','070.png',
              '071.png','072.png','073.png','074.png','075.png','076.png','077.png','078.png','079.png','080.png',
              '081.png','082.png','083.png','084.png','085.png','086.png','087.png','088.png','089.png','090.png',
              '091.png','092.png','093.png','094.png','095.png','096.png','097.png','098.png','099.png','100.png',
              '101.png','102.png','103.png','104.png','105.png','106.png','107.png','108.png','109.png','100.png',
              '111.png','112.png','113.png','114.png','115.png','116.png','117.png','118.png','119.png','120.png',
              '121.png','122.png','123.png','124.png','125.png','126.png','127.png','128.png','129.png','130.png',
              '131.png','132.png','133.png','134.png','135.png','136.png','137.png','138.png','139.png','140.png',
              '141.png','142.png','143.png','144.png','145.png','146.png','147.png','148.png','149.png','150.png',
              '151.png','152.png','153.png','154.png','155.png','156.png','157.png','158.png','159.png','160.png'
              ]

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)
    

def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb
    
def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
    
def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)
        
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)
        
    if np.random.random() < 0.25:
        xb = blur(xb)
    
    if np.random.random() < 0.2:
        xb = add_noise(xb)
        
    return xb,yb

def creat_dataset(image_num = 16000, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread('E:/python3.6_work/unet/data16000/images160_png/' + image_sets[i])  # 3 channels
        label_img = cv2.imread('E:/python3.6_work/unet/data16000/gt160_png/' + image_sets[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
#            cv2.imwrite(('E:/python3.6_work/unet/data160000/yuan_src_roi/%d.png' % g_count),src_roi)
#            cv2.imwrite(('E:/python3.6_work/unet/data160000/yuan_label_roi/%d.png' % g_count),label_roi)
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
            
            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_roi *50
            
            cv2.imwrite(('E:/python3.6_work/unet/data16000/visualize/%d.png' % g_count),visualize)
            cv2.imwrite(('E:/python3.6_work/unet/data16000/src/%d.png' % g_count),src_roi)
            cv2.imwrite(('E:/python3.6_work/unet/data16000/label/%d.png' % g_count),label_roi)
            count += 1 
            g_count += 1


            
    

if __name__=='__main__':  
    creat_dataset(mode='augment')