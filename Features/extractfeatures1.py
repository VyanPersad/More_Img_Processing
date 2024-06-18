#improvement on the segmentation (actually done really well) on color analysis alone

import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor

def getGLCMFeatures(img):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img, distances, angles, levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    return features

def calculate_features(img, mask):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the mask to match the dimensions of the gray image
    mask_resized = cv2.resize(mask, (gray_image.shape[1], gray_image.shape[0]))
    
    # Calculate GLCM
    masked_gray = gray_image[mask_resized == 255]
    glcm_features = getGLCMFeatures(masked_gray.reshape(masked_gray.shape[0], 1))
    
    # Calculate Gabor features
    frequencies = [0.1, 0.3, 0.5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []
    for frequency in frequencies:
        for theta in angles:
            kernel = np.real(gabor(gray_image, frequency, theta=theta))
            # Resize the kernel to match the dimensions of the mask
            kernel_resized = cv2.resize(kernel, (mask.shape[1], mask.shape[0]))
            gabor_features.append(np.mean(kernel_resized[mask == 255]))
    
    # Calculate K value
    b, g, r = cv2.split(img)
    c = 255 - r
    m = 255 - g
    y = 255 - b
    k = np.minimum(np.minimum(c, m), y)
    k_value = np.mean(k[mask_resized == 255])
    
    return glcm_features, gabor_features, k_value


# Directory containing images
image_directory = 'CroppedImgs/'

# CSV file path
csv_file_path = 'data.csv'
header_names = ['Basename','Contrast Hyper', 'Dissimilarity Hyper','Homogeneity Hyper','Energy Hyper','Correlation Hyper', 'ASM Hyper', 
                'Gabor1 Hyper','Gabor2 Hyper','Gabor3 Hyper','Gabor4 Hyper','Gabor5 Hyper','Gabor6 Hyper','Gabor7 Hyper','Gabor8 Hyper',
                'Gabor9 Hyper','Gabor10 Hyper','Gabor11 Hyper','k-Value Hyper', 'GLCM Contrast Normal', 'GLCM Dissimilarity Normal',
                'GLCM Homogeneity Normal', 'GLCM ASM Normal', 'GLCM Energy Normal', 'GLCM Correlation Normal',
                'Gabor1 Normal', 'Gabor2 Normal', 'Gabor3 Normal', 'Gabor4 Normal', 'Gabor5 Normal', 'Gabor6 Normal',
                'Gabor7 Normal', 'Gabor8 Normal', 'Gabor9 Normal', 'Gabor10 Normal', 'Gabor11 Normal', 'k-Value Normal']

# Define header names for differences
difference_header_names = ['Contrast Difference', 'Dissimilarity Difference', 'Homogeneity Difference',
                           'Energy Difference', 'Correlation Difference', 'ASM Difference',
                           'Gabor1 Difference', 'Gabor2 Difference', 'Gabor3 Difference',
                           'Gabor4 Difference', 'Gabor5 Difference', 'Gabor6 Difference',
                           'Gabor7 Difference', 'Gabor8 Difference', 'Gabor9 Difference',
                           'Gabor10 Difference', 'Gabor11 Difference', 'k-Value Difference']

# Check if the CSV file exists
file_exists = os.path.exists(csv_file_path)

# Iterate over images in the directory
for file in os.listdir(image_directory):
    base_name = file.split("_")[0]
    file_path = os.path.join(image_directory, file)
    img = cv2.imread(file_path)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean pixel value
    mean_val = np.mean(gray_img)
    
    # Create masks based on the mean value
    normal_mask = np.uint8(gray_img > mean_val) * 255  # Normal region is darker
    hyper_mask = np.uint8(gray_img <= mean_val) * 255    # Hyperpigmented region is lighter
    
    # Apply the hyperpigmented mask to the original image to keep the colors of the hyperpigmented regions
    hyper_colored = cv2.bitwise_and(img, img, mask=hyper_mask)
    # Apply the normal mask to the original image to keep the colors of the normal regions
    normal_colored = cv2.bitwise_and(img, img, mask=normal_mask)

    # Visualize the original image and the masked regions
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[1].imshow(cv2.cvtColor(normal_colored, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Normal Region')
    ax[2].imshow(cv2.cvtColor(hyper_colored, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Hyperpigmented Region')
    plt.show()
    
    # Calculate features for normal region
    normal_glcm_features, normal_gabor_features, normal_k_value = calculate_features(img, normal_mask)
    
    # Calculate features for hyperpigmented region
    hyper_glcm_features, hyper_gabor_features, hyper_k_value = calculate_features(img, hyper_mask)
    
    # Calculate differences
    glcm_differences = [round(hyper_feature - normal_feature, 4) for hyper_feature, normal_feature in zip(hyper_glcm_features, normal_glcm_features)]
    gabor_differences = [round(hyper_gabor_feature - normal_gabor_feature, 4) for hyper_gabor_feature, normal_gabor_feature in zip(hyper_gabor_features, normal_gabor_features)]
    k_value_difference = round(hyper_k_value - normal_k_value, 4)

    # Combine differences
    differences = glcm_differences + gabor_differences + [k_value_difference]

    # Write data to CSV
    data = [{'Basename': base_name,
             'Contrast Hyper': hyper_glcm_features[0], 'Dissimilarity Hyper': hyper_glcm_features[1], 
             'Homogeneity Hyper': hyper_glcm_features[2], 'Energy Hyper': hyper_glcm_features[3], 
             'Correlation Hyper': hyper_glcm_features[4], 'ASM Hyper': hyper_glcm_features[5],
             'Gabor1 Hyper': round(hyper_gabor_features[0], 4), 'Gabor2 Hyper': round(hyper_gabor_features[1], 4),
             'Gabor3 Hyper': round(hyper_gabor_features[2], 4), 'Gabor4 Hyper': round(hyper_gabor_features[3], 4),
             'Gabor5 Hyper': round(hyper_gabor_features[4], 4), 'Gabor6 Hyper': round(hyper_gabor_features[5], 4),
             'Gabor7 Hyper': round(hyper_gabor_features[6], 4), 'Gabor8 Hyper': round(hyper_gabor_features[7], 4),
             'Gabor9 Hyper': round(hyper_gabor_features[8], 4), 'Gabor10 Hyper': round(hyper_gabor_features[9], 4),
             'Gabor11 Hyper': round(hyper_gabor_features[10], 4), 'k-Value Hyper': round(hyper_k_value, 4),
             'GLCM Contrast Normal': normal_glcm_features[0], 'GLCM Dissimilarity Normal': normal_glcm_features[1],
             'GLCM Homogeneity Normal': normal_glcm_features[2], 'GLCM ASM Normal': normal_glcm_features[3],
             'GLCM Energy Normal': normal_glcm_features[4], 'GLCM Correlation Normal': normal_glcm_features[5],
             'Gabor1 Normal': round(normal_gabor_features[0], 4), 'Gabor2 Normal': round(normal_gabor_features[1], 4),
             'Gabor3 Normal': round(normal_gabor_features[2], 4), 'Gabor4 Normal': round(normal_gabor_features[3], 4),
             'Gabor5 Normal': round(normal_gabor_features[4], 4), 'Gabor6 Normal': round(normal_gabor_features[5], 4),
             'Gabor7 Normal': round(normal_gabor_features[6], 4), 'Gabor8 Normal': round(normal_gabor_features[7], 4),
             'Gabor9 Normal': round(normal_gabor_features[8], 4), 'Gabor10 Normal': round(normal_gabor_features[9], 4),
             'Gabor11 Normal': round(normal_gabor_features[10], 4), 'k-Value Normal': round(normal_k_value, 4),
             'Contrast Difference': glcm_differences[0], 'Dissimilarity Difference': glcm_differences[1],
             'Homogeneity Difference': glcm_differences[2], 'Energy Difference': glcm_differences[3],
             'Correlation Difference': glcm_differences[4], 'ASM Difference': glcm_differences[5],
             'Gabor1 Difference': gabor_differences[0], 'Gabor2 Difference': gabor_differences[1],
             'Gabor3 Difference': gabor_differences[2], 'Gabor4 Difference': gabor_differences[3],
             'Gabor5 Difference': gabor_differences[4], 'Gabor6 Difference': gabor_differences[5],
             'Gabor7 Difference': gabor_differences[6], 'Gabor8 Difference': gabor_differences[7],
             'Gabor9 Difference': gabor_differences[8], 'Gabor10 Difference': gabor_differences[9],
             'Gabor11 Difference': gabor_differences[10], 'k-Value Difference': k_value_difference
            }]

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_names + difference_header_names)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()
        
        # Write rows
        for row in data:
            writer.writerow(row)


    print("Length of base_name:", len([base_name]))
    print("Length of hyper_glcm_features:", len(hyper_glcm_features))
    print("Length of hyper_gabor_features:", len(hyper_gabor_features))
    print("Length of hyper_k_value:", len([hyper_k_value]))
    print("Length of normal_glcm_features:", len(normal_glcm_features))
    print("Length of normal_gabor_features:", len(normal_gabor_features))
    print("Length of normal_k_value:", len([normal_k_value]))
    print("Length of glcm_differences:", len(glcm_differences))
    print("Length of gabor_differences:", len(gabor_differences))
    print("Length of k_value_difference:", len([k_value_difference]))

    print("Base name:", [base_name])
    print("Hyper GLCM features:", list(hyper_glcm_features))
    print("Hyper Gabor features:", list(hyper_gabor_features))
    print("Hyper k value:", [hyper_k_value])
    print("Normal GLCM features:", list(normal_glcm_features))
    print("Normal Gabor features:", list(normal_gabor_features))
    print("Normal k value:", [normal_k_value])
    print("GLCM differences:", glcm_differences)
    print("Gabor differences:", gabor_differences)
    print("k value difference:", [k_value_difference])
