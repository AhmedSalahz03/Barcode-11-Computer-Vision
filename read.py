# Barcode Detection using OpenCV (Code 11 Symbology)
import cv2 as cv
import numpy as np

img = cv.imread('Samples/image1.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('Original Image', img)

median_blurred_img = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median_blurred_img)

# Binarize the image using a threshold
_, binary_img = cv.threshold(median_blurred_img, 128, 255, cv.THRESH_BINARY_INV)
cv.imshow('Binarized Image', binary_img)

# Apply morphological closing to reduce gaps
kernel = np.ones((5, 5), np.uint8)
binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
cv.imshow('Morphologically Closed Image', binary_img)

# Find contours in the binarized image
contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the barcode
largest_contour = max(contours, key=cv.contourArea)

# Get the bounding box of the largest contour
x, y, w, h = cv.boundingRect(largest_contour)

# Crop the image to the bounding box
cropped_img = median_blurred_img[y:y+h, x:x+w]
cv.imshow('Cropped Image', cropped_img)

cv.waitKey(0)
cv.destroyAllWindows()