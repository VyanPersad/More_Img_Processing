import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from statistics import mean
from Functions.fileFunctions import *
from Functions.maskFunctions import HSVskinMask
from Functions.ImgAnalysisFunctions import k_means

img_title = ['Image','k-Means','Hyper','Normal']
filepath = 'Manual_Crop'
#file = 'IR018.jpg'

def k_meansFolder(filepath):
    for file in os.listdir(filepath):
        imagebgr = readFromFile(filepath, file)

        crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
        segmented_crppd, centre = k_means(crppd_img_HSV_1,2)

        imgArray = [] 
        #Builds the image array.
        imgArray.append(imagebgr)
        #imgArray.append(crppd_img_HSV_1)
        imgArray.append(segmented_crppd)
        centArr = []
        for i in range(3):          
            if (mean([centre[i][0], centre[i][1], centre[i][2]]) > 10): 
                centArr.append([centre[i][0], centre[i][1], centre[i][2]])

        hyper = np.zeros((1, 1, 3), dtype=np.uint8)       
        normal = np.zeros((1, 1, 3), dtype=np.uint8)       

        if (mean([centArr[0][0], centArr[0][1], centArr[0][2]])<mean([centArr[1][0], centArr[1][1], centArr[1][2]])):  
            centreDeets = f' C1-BGR-Hyper-{centArr[0][0], centArr[0][1], centArr[0][2]} \
                \n C2-BGR-Normal-{centArr[1][0], centArr[1][1], centArr[1][2]}'
            hyper[0, 0] = (centArr[0][0], centArr[0][1], centArr[0][2])
            normal[0, 0] = (centArr[1][0], centArr[1][1], centArr[1][2])
            imgArray.append(hyper)
            imgArray.append(normal)

        elif(mean([centArr[0][0], centArr[0][1], centArr[0][2]])>mean([centArr[1][0], centArr[1][1], centArr[1][2]])):
            centreDeets = f' C1-BGR-Hyper-{centArr[1][0], centArr[1][1], centArr[1][2]} \
                \n C2-BGR-Normal-{centArr[0][0], centArr[0][1], centArr[0][2]}'
            hyper[0, 0] = (centArr[1][0], centArr[1][1], centArr[1][2])
            normal[0, 0] = (centArr[0][0], centArr[0][1], centArr[0][2])
            imgArray.append(hyper)
            imgArray.append(normal)

        # Display the image
        #showfilmStripPlot(img_title, imgArray, 5, centreDeets, f'{file.split(".")[0]}')
        filmStripPlot(img_title,imgArray,4,'k-means_M_Crp',file)

def k_meansFile(filepath, file):
    imagebgr = readFromFile(filepath, file)

    crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
    segmented_crppd, centre = k_means(crppd_img_HSV_1)

    imgArray = [] 
    #Builds the image array.
    imgArray.append(imagebgr)
    imgArray.append(crppd_img_HSV_1)
    imgArray.append(segmented_crppd)
    for i in range(3):
        image_pyplot = np.zeros((1, 1, 3), dtype=np.uint8)
        if (mean([centre[i][0], centre[i][1], centre[i][2]]) > 10):
            image_pyplot[0, 0] = (centre[i][0], centre[i][1], centre[i][2]) 
            imgArray.append(image_pyplot)
    centArr = []
    # Centre Details
    for i in range(3):
        if (mean([centre[i][0], centre[i][1], centre[i][2]]) > 10):
            centArr.append([centre[i][0], centre[i][1], centre[i][2]])

    if (mean([centArr[0][0], centArr[0][1], centArr[0][2]])<mean([centArr[1][0], centArr[1][1], centArr[1][2]])):        
        centreDeets = f' C1-BGR-Hyper-{centArr[0][0], centArr[0][1], centArr[0][2]} \
        \n C2-BGR-Normal-{centArr[1][0], centArr[1][1], centArr[1][2]}'

    elif(mean([centArr[0][0], centArr[0][1], centArr[0][2]])>mean([centArr[1][0], centArr[1][1], centArr[1][2]])):
        centreDeets = f' C1-BGR-Hyper-{centArr[1][0], centArr[1][1], centArr[1][2]} \
        \n C2-BGR-Normal-{centArr[0][0], centArr[0][1], centArr[0][2]}'
    # Display the image
    showfilmStripPlot(img_title, imgArray, 5, centreDeets, f'{file.split(".")[0]}')
    #filmStripPlot(img_title,imgArray,5,'k-means_filmStrips',file)


k_meansFolder(filepath)