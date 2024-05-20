from skimage.feature import graycomatrix
from skimage.feature import graycoprops
import os
import cv2
import numpy as np
import csv

for file in os.listdir('CroppedImgs/'):
        
    file_path = f'CroppedImgs/{file}'
    img = cv2.imread(file_path)
    base_name = file.split("_")[0]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #Calculate GLCM with specified parameters
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for pixel pairs
    levels = 256  # Number of gray levels
    symmetric = True
    normed = True
            
    glcm = graycomatrix(gray_image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)        
        
    cont = round(graycoprops(glcm, 'contrast').ravel()[0], 4)
    diss = round(graycoprops(glcm, 'dissimilarity').ravel()[0], 4)
    homo = round(graycoprops(glcm, 'homogeneity').ravel()[0], 4)
    ener = round(graycoprops(glcm, 'energy').ravel()[0], 4)
    corr = round(graycoprops(glcm, 'correlation').ravel()[0], 4)
    asm = round(graycoprops(glcm, 'ASM').ravel()[0], 4)
            
    data = [{'Basename':base_name,'Contrast': cont, 
             'Dissimilarity': diss, 'Homogeneity':homo, 
             'Energy':ener, 'Correlation':corr, 'ASM':asm}]

    header_names = ['Basename','Contrast', 'Dissimilarity','Homogeneity','Energy','Correlation','ASM']

    csv_file_path = 'data.csv'
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_names)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()
        
        # Write rows
        for row in data:
            writer.writerow(row)
