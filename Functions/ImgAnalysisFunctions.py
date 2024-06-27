import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def rgbHist(image):
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
    #[0,1,2] = [R,G,B]
    return [imgCol, rgbArr]

def plotrgbHist(rgbArray):
    colorArr = ['r','g','b']
    for i in range(3):
        plt.plot(rgbArray[i], color=colorArr[i])

    plt.tight_layout
    return plt.show()

def k_means(image):
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