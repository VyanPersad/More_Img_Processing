import os
import cv2
import numpy as np
import csv
import random
import math
from PIL import Image
import matplotlib.pyplot as plt

def getRandoPxs(file_path, num_pxs=2):
    img = Image.open(file_path)
    width, height = img.size

    def rejectblk_whit(pxl):
        r,g,b = pxl
        return r==g==b ==0 or r==b==g==255
    
    selected_pxls=[]
    while len(selected_pxls)<num_pxs:
        x,y = random.randint(0,width-1), random.randint(0,height-1)
        pxl_val = img.getpixel((x,y))
        if not rejectblk_whit(pxl_val):
            selected_pxls.append(pxl_val)

    return selected_pxls

def calc_bright(pixel):
    r,g,b = pixel
    return 0.299*r+0.587*g+0.114*b

def light_dark(pixel_list):
    normal = None
    hyper = None
    max_lum = float("-inf")
    min_lum = float("inf")
    for pxl in pixel_list:
        lum = calc_bright(pxl)
        if lum > max_lum:
            max_lum = lum
            normal = pxl
        if lum < min_lum:
            min_lum = lum
            hyper = pxl

    return normal, hyper        

def avg_colour(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return avg_color

def rounded(r,g,b):
    rr=round(r,2)
    rg=round(g,2)
    rb=round(b,2)

    return rr,rg,rb

def write_to_csv(output_file_path, fieldnames, data):
    
    file_exists = os.path.exists(output_file_path)

    with open(output_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()
        # Write rows
        for row in data:
            writer.writerow(row)
    
#0<--Blk+++White-->255
img_files_normal_otsu = []
img_files_adaptive_otsu = []
img_title = ['Img','Binary','Normal','Hyper']
#f, plot = plt.subplots(5, 4)
f, customPlot = plt.subplots(2, 2)
# plt.subplot(row, column)

'''
      0 1 2 3 
    0 x x x x  0
    1 x x x x  4
    2 x x x x  8
         
'''

for file in os.listdir('CroppedImgs/'):
        
    file_path = f'CroppedImgs/{file}'
    #print(file_path)
    img = cv2.imread(file_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # otsu threshold seperation of hyper and normal pigmentation   
    # otsu with adaptive thresholding
    # The block_size determines the overall area around the pixel over
    # which the average or weighted average will be calculated.
    # A larger block is good for regions with lighting changes.
    # A smaller block is good for high or fine detail regions.
    # The C param thast is sibtracted from the average
    # Allows for adjustement of the threshold value based on brightness.
    # C +ve -> conservative threshold, shift toward darker region
    # C -ve -> permissive threshold, shift toward lighter region
    otsu_threshold, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block_size = 11
    C = 0
    binary_image_w_adapt = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    normal = cv2.bitwise_and(img, img, mask=binary_image)
    hyper = cv2.bitwise_and(img, img, mask=~binary_image)

    normal_a = cv2.bitwise_and(img, img, mask=binary_image_w_adapt)
    hyper_a = cv2.bitwise_and(img, img, mask=~binary_image_w_adapt)
    # This builds the number of columns. 
    img_files_normal_otsu.append(img)
    img_files_normal_otsu.append(binary_image)
    img_files_normal_otsu.append(normal)
    img_files_normal_otsu.append(hyper)

    img_files_adaptive_otsu.append(img)
    img_files_adaptive_otsu.append(binary_image_w_adapt)
    img_files_adaptive_otsu.append(normal_a)
    img_files_adaptive_otsu.append(hyper_a)

'''
for img in img_files_normal_otsu:
    # This is for the number of rows we have to iterate through
    for i in range(0,5):
        if (i==0):
            x=0
        elif (i==1):
            x=4
        elif (i==2):
            x=8
        elif (i==3):
            x=12
        elif (i==4):
            x=16        
        # This governed by the no. of columns we have to iterate through    
        for j in range(0, 4):
            plot[i,j].imshow(img_files_normal_otsu[j+x])
            plot[i,j].set_title(img_title[j])

    plt.show()

for img in img_files_adaptive_otsu:
    # This is for the number of rows we have to iterate through
    for i in range(0,5):
        if (i==0):
            x=0
        elif (i==1):
            x=4
        elif (i==2):
            x=8
        elif (i==3):
            x=12
        elif (i==4):
            x=16        
        # This governed by the no. of columns we have to iterate through    
        for j in range(0, 4):
            plot[i,j].imshow(img_files_adaptive_otsu[j+x])
            plot[i,j].set_title(img_title[j])

    plt.show()



for img in img_files_adaptive_otsu:
    customPlot[0,0].imshow(img_files_adaptive_otsu[2])
    customPlot[0,0].set_title(img_title[2])
    customPlot[0,1].imshow(img_files_adaptive_otsu[3])
    customPlot[0,1].set_title(img_title[3])

    plt.show()
'''
for img in img_files_normal_otsu:
    customPlot[0,0].imshow(img_files_normal_otsu[2])
    customPlot[0,0].set_title(img_title[2])
    customPlot[0,1].imshow(img_files_normal_otsu[3])
    customPlot[0,1].set_title(img_title[3])


plt.show()    