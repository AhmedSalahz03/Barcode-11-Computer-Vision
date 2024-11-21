import cv2 as cv
import numpy as np

from crop import crop_barcode

# Load the image
img = cv.imread('Samples/image1.jpg', cv.IMREAD_GRAYSCALE)

# Resize the image to increase its dimensions
scale_factor = 4  # Adjust the scale factor as needed
resized_img = cv.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor))

# Binarize the image using a threshold
_, binary_img = cv.threshold(resized_img, 128, 255, cv.THRESH_BINARY_INV)
cv.imshow('Binarized Image', binary_img)

kernel_height1 = 17
kernel_width1 = 17  
kernel1 = np.zeros((kernel_height1, kernel_width1), np.uint8)
kernel1[:, kernel_width1 // 2] = 1  

binary_img1 = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel1)
cv.imshow('Morphologically Opened Image', binary_img1)

kernel_height1 = 21  
kernel_width1 = 21 
kernel1 = np.zeros((kernel_height1, kernel_width1), np.uint8)
kernel1[:, kernel_width1 // 2] = 1  

binary_img = cv.morphologyEx(binary_img1, cv.MORPH_CLOSE, kernel1)
cv.imshow('Morphologically Closed Image', binary_img)

kernel_height2 = 1 
kernel_width2 = 5  
kernel2 = np.ones((kernel_height2, kernel_width2), np.uint8) 

binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel2)

cropped_img = crop_barcode(binary_img)

# Create a vertical kernel (all 1's vertically)
kernel_vertical = np.ones((17, 1), np.uint8)  # Adjust the height as needed

# Apply dilation with the vertical kernel
dilated_img = cv.dilate(cropped_img, kernel_vertical)
cv.imshow('Dilated Image', dilated_img)
final_img = cv.morphologyEx(cropped_img, cv.MORPH_CLOSE, kernel1)

# Invert 
inverted_img = cv.bitwise_not(final_img)
cv.imshow('Inverted Morphologically Closed Image', inverted_img)



cv.waitKey(0)
cv.destroyAllWindows()