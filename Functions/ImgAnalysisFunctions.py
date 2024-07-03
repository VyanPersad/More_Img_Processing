import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from statistics import mean
from Functions.maskFunctions import *
from Functions.fileFunctions import *

def k_means(image, k=3):
    pxl_val = image.reshape((-1,3))
    pxl_val = np.float32(pxl_val)
    #print(pxl_val.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, (centers) = cv2.kmeans(pxl_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    #Remember the center are output as bgr
    return segmented_image, centers

def k_meansFolder(filepath, destFolderPath):
    for file in os.listdir(filepath):
        imagebgr = readFromFile(filepath, file)

        crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
        segmented_crppd, centre = k_means(crppd_img_HSV_1,2)
        img_title = ['Image','k-Means','Hyper','Normal']
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
        filmStripPlot(img_title,imgArray,4,f'{destFolderPath}',file)

def k_meansFile(filepath, file):
    imagebgr = readFromFile(filepath, file)

    crppd_img_HSV_1 = HSVskinMask(imagebgr,[20,255,255],[3,15,10])
    segmented_crppd, centre = k_means(crppd_img_HSV_1)
    img_title = ['Image','k-Means','Hyper','Normal']
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
    showfilmStripPlot(img_title, imgArray, len(img_title), centreDeets, f'{file.split(".")[0]}')
    #filmStripPlot(img_title,imgArray,len(img_title),'k-means_filmStrips',file)

def grayHist(image):
    imgGr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayHist = cv2.calcHist([imgGr],[0],None,[256],[0,256])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgGr = cv2.cvtColor(imgGr, cv2.COLOR_BGR2RGB)

    return [image, imgGr, grayHist]

def grayHistnoBlk(image):

    imgGr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    msk = (imgGr >0).astype(np.uint8) 
    #This filters the wholly black pixels.
    #That is those with a value of 0
    grayHist = cv2.calcHist([imgGr],[0],msk,[256],[0,256])

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #imgGr = cv2.cvtColor(imgGr, cv2.COLOR_BGR2RGB)

    return [image, imgGr, grayHist]

def rgbHist_w_Msk(image):
    rgbArr = []
    upper=[20,255,255] 
    lower=[3,15,10]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    for i in range(3):
        hist = cv2.calcHist([image],[i],skinMask,[256],[0,256])
        rgbArr.append(hist)
    
    imgCol = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #The second term is an array
    #[0,1,2] = [r,g,b]
    return [imgCol, rgbArr]

def rgbHist(image):
    rgbArr = []

    for i in range(3):
        hist = cv2.calcHist([image],[i],None,[256],[0,256])
        rgbArr.append(hist)
    
    imgCol = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #The second term is an array
    #[0,1,2] = [r,g,b]
    return [imgCol, rgbArr]

def plotrgbHist(rgbArray, title, xLabel, yLabel):
    colorArr = ['r','g','b']
    for i in range(3):
        plt.plot(rgbArray[i], color=colorArr[i])
    plt.title(title)
    plt.set_xlabel(xLabel)
    plt.set_ylabel(yLabel)
    plt.tight_layout
    return plt.show()

def plotrgbCustom(customPlot, rgbHist):
    colorArr = ['r','g','b']
    for i in range(3):
        customPlot[i].plot(rgbHist[1][i], color=colorArr[i]) 

def contrastFolderLoop(filepath, destFilePath):
  for file in os.listdir(filepath):
      image = readFromFile(filepath, file)

      HSV_Contrast = []
      HSV_Contrast.append(image)
      alpha = [1, 1.5, 2.0, 2.5]
      
      for i in range(4):
          contrasted_img = cv2.convertScaleAbs(image, alpha=alpha[i])
          HSV_Contrast.append(HSVskinMask(contrasted_img))

      img_title = ['Normal','Contrast a = 1','Contrast a = 1.5','Contrast a = 2.0','Contrast a = 2.5']
      
      f, customPlot = plt.subplots(1, 5, figsize=(10,5))
      base_name = file.split(".")[0]

      for i in range(5):
        customPlot[i].set_title(img_title[i])
        for j in range(5):
          customPlot[j].imshow(cv2.cvtColor(HSV_Contrast[j], cv2.COLOR_BGR2RGB))

      plt.tight_layout()
      
      makeFolder(destFilePath)
      #plt.show()
      plt.savefig(f'{destFilePath}/ContrastImgStrip_{base_name}.png')
      plt.close(f)

def contrastAnalysis(filepath, file, destFilePath):
  image = readFromFile(filepath, file)

  HSV_Contrast = []
  HSV_Contrast.append(image)
  alpha = [1, 1.5, 2.0, 2.5]
  
  for i in range(4):
      contrasted_img = cv2.convertScaleAbs(image, alpha=alpha[i])
      HSV_Contrast.append(HSVskinMask(contrasted_img))

  img_title = ['Normal','Contrast a = 1','Contrast a = 1.5','Contrast a = 2.0','Contrast a = 2.5']
  
  f, customPlot = plt.subplots(1, 5, figsize=(10,5))
  base_name = file.split(".")[0]

  for i in range(5):
    customPlot[i].set_title(img_title[i])
    for j in range(5):
      customPlot[j].imshow(cv2.cvtColor(HSV_Contrast[j], cv2.COLOR_BGR2RGB))

  plt.tight_layout()
  
  makeFolder(destFilePath)
  plt.show()
  #plt.savefig(f'{destFilePath}/ContrastImgStrip_{base_name}.png')
  plt.close(f)

def cropFolder(filepath, destFilePath):
    for file in os.listdir(filepath):
        image = readFromFile(filepath, file)

        crppd_img_HSV_1 = HSVskinMask(image,[20,255,255],[3,15,10])
        crppd_img_HSV_2 = HSVskinMask(image,[25,255,255],[0,40,0])

        crppd_img_YCb = YCbCrskinMask(image,[255,173,133],[0,138,67])
        #This way only the 4 imgs for this one image is stored. 
        imgArray = []
        imgArray.append(image)
        imgArray.append(crppd_img_HSV_1)
        imgArray.append(crppd_img_HSV_2)
        imgArray.append(crppd_img_YCb)

        #hsvCrp.append(crppd_img_HSV)
        #ycbcrCrp.append(crppd_img_YCb)

        writeImgTofile(file, 'CrppdImg_HSV_Set_1', 'HSV_1_C', 'png', crppd_img_HSV_1)
        writeImgTofile(file, 'CrppdImg_HSV_Set_2', 'HSV_2_C', 'png', crppd_img_HSV_2)

        writeImgTofile(file, 'CrppdImg_YCrCb', 'YCrCb_C', 'png', crppd_img_YCb)

        f, customPlot = plt.subplots(1, 4, figsize=(15,5))
        img_title = ['Normal','HSV_1','HSV_2','YCrCb']
        base_name = file.split(".")[0]

        #The number corresponds to the no. of imgs in the film strip
        for i in range(4):
            customPlot[i].set_title(img_title[i])
            #cv2 works in BGR plt works in RGB hence the conversion 
            customPlot[i].imshow(cv2.cvtColor(imgArray[i], cv2.COLOR_BGR2RGB))
            
        plt.tight_layout()
        
        makeFolder(destFilePath)
        #plt.show()
        plt.savefig(f'{destFilePath}/ImgStrip_{base_name}_Crppd.png')
        plt.close(f)

def cropFile(filepath, file, destFilePath):
  
  image = readFromFile(filepath, file)

  crppd_img_HSV_1 = HSVskinMask(image,[20,255,255],[3,15,10])
  crppd_img_HSV_2 = HSVskinMask(image,[25,255,255],[0,40,0])

  crppd_img_YCb = YCbCrskinMask(image,[255,173,133],[0,138,67])
  #This way only the 4 imgs for this one image is stored. 
  imgArray = []
  imgArray.append(image)
  imgArray.append(crppd_img_HSV_1)
  imgArray.append(crppd_img_HSV_2)
  imgArray.append(crppd_img_YCb)

  #hsvCrp.append(crppd_img_HSV)
  #ycbcrCrp.append(crppd_img_YCb)

  writeImgTofile(file, 'CrppdImg_HSV_Set_1', 'HSV_1_C', 'png', crppd_img_HSV_1)
  writeImgTofile(file, 'CrppdImg_HSV_Set_2', 'HSV_2_C', 'png', crppd_img_HSV_2)
  writeImgTofile(file, 'CrppdImg_YCrCb', 'YCrCb_C', 'png', crppd_img_YCb)

  f, customPlot = plt.subplots(1, 4, figsize=(15,5))
  img_title = ['Normal','HSV_1','HSV_2','YCrCb']
  base_name = file.split(".")[0]

  #The number corresponds to the no. of imgs in the film strip
  for i in range(4):
    customPlot[i].set_title(img_title[i])
    #cv2 works in BGR plt works in RGB hence the conversion 
    customPlot[i].imshow(cv2.cvtColor(imgArray[i], cv2.COLOR_BGR2RGB))
    
  plt.tight_layout()
  
  makeFolder(destFilePath)
  plt.show()
  #plt.savefig(f'{destFilePath}/ImgStrip_{base_name}.png')
  plt.close(f)

def imgAnalysisFile1x4(filepath, file, saveState, destFolderPath = None):
    image = readFromFile(filepath, file)

    grayHistogram = grayHistnoBlk(image)
    rgbHistogram = rgbHist_w_Msk(image)
    kmeansImg = k_means(image)

    ImgTitles = ['Image','k-Means','Gray Hist','RGB Hist']

    base_name = file.split(".")[0]
    f, filmPlot = plt.subplots(1,4, figsize=(10,5))

    plt.suptitle(f'{base_name}', x=0.05, y=0.9, ha='left', va='top')
    for i in range (len(ImgTitles)):
        filmPlot[i].set_title(ImgTitles[i])
        if (i == 0 ):
            filmPlot[i].imshow(cv2.cvtColor(grayHistogram[i], cv2.COLOR_BGR2RGB))
        elif (i == 1 ):
            filmPlot[i].imshow(cv2.cvtColor(kmeansImg[0], cv2.COLOR_BGR2RGB))    
        elif (i==2):
            filmPlot[i].set_ylabel('Gray')
            filmPlot[i].plot(grayHistogram[i], color='black')
        elif (i==3):
            colorArr = ['r','g','b']
            for j in range(3):
                filmPlot[i].set_ylabel('RGB')
                filmPlot[i].plot(rgbHistogram[1][j], color=colorArr[j])    
    plt.tight_layout()
    if (destFolderPath == None):
        if (saveState == 0):
            plt.show()
    else:    
        makeFolder(destFolderPath)
    if (saveState == 0):
        plt.show()
    elif (saveState == 1):    
        plt.savefig(f'{destFolderPath}/{base_name}_ImgAnalysis_fStrip.png')
    plt.close(f)

def imgAnalysisFile1x3(filepath, file, saveState, destFolderPath = None):
    image = readFromFile(filepath, file)

    grayHistogram = grayHistnoBlk(image)
    rgbHistogram = rgbHist(image)
    kmeansImg = k_means(image)

    ImgTitles = ['Image','Crppd Gray','Combined Hist']

    base_name = file.split(".")[0]
    f, filmPlot = plt.subplots(1,len(ImgTitles), figsize=(10,5))
    plt.suptitle(f'{base_name}', x=0.05, y=0.9, ha='left', va='top')
    for i in range (len(ImgTitles)):
        colorArr = ['r','g','b']
        filmPlot[i].set_title(ImgTitles[i])
        filmPlot[2].set_ylabel('RGB')
        filmPlot[2].plot(rgbHistogram[1][i], color=colorArr[i])
        if (i == 0):
            filmPlot[i].imshow(cv2.cvtColor(grayHistogram[0], cv2.COLOR_BGR2RGB))
        elif (i == 1):
            filmPlot[i].imshow(cv2.cvtColor(kmeansImg[0], cv2.COLOR_BGR2RGB))
        elif (i==2):
            filmPlot[i] = filmPlot[i].twinx()
            filmPlot[i].set_ylabel('Gray')
            filmPlot[i].plot(grayHistogram[2], color='black')
  
    plt.tight_layout()
    if (destFolderPath == None):
        if (saveState == 0):
            plt.show()
    else:    
        makeFolder(destFolderPath)
    if (saveState == 0):
        plt.show()
    elif (saveState == 1):    
        plt.savefig(f'{destFolderPath}/Crp_fStrip_{base_name}.png')
    plt.close(f)

def imgAnalysisFolder(filepath, saveState, destFolderPath = None):
    for file in os.listdir(filepath):
        image = readFromFile(filepath, file)

        grayHistogram = grayHistnoBlk(image)
        rgbHistogram = rgbHist(image)

        ImgTitles = ['Image','Crppd Gray','Gray Hist','RGB Hist']

        base_name = file.split(".")[0]
        f, filmPlot = plt.subplots(1,4, figsize=(10,5))
        plt.suptitle(f'{base_name}', x=0.05, y=0.9, ha='left', va='top')
        for i in range (len(ImgTitles)):
            filmPlot[i].set_title(ImgTitles[i])
            if (i < 2):
                filmPlot[i].imshow(cv2.cvtColor(grayHistogram[i], cv2.COLOR_BGR2RGB))
            elif (i==2):
                filmPlot[i].plot(grayHistogram[2], color='black')
            elif (i==3):
                colorArr = ['r','g','b']
                for j in range(3):
                    filmPlot[i].plot(rgbHistogram[1][j], color=colorArr[j])    
        
    plt.tight_layout()
    if (destFolderPath == None):
        if (saveState == 0):
            plt.show()
    else:    
        makeFolder(destFolderPath)
    if (saveState == 0):
        plt.show()
    elif (saveState == 1):    
        plt.savefig(f'{destFolderPath}/Img_Strip_{base_name}.png')
        
    plt.close(f)
