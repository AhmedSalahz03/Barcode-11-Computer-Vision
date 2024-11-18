import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Samples/image1.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('Original Image', img)

# Apply median blur to reduce noise
#median_blurred_img = cv.medianBlur(img, 3)
#cv.imshow('Median Blur', median_blurred_img)

#denoised_img = cv.fastNlMeansDenoising(median_blurred_img, None, 30, 7, 21)

# Find contours in the binary image
contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Assume the largest contour corresponds to the barcode
largest_contour = max(contours, key=cv.contourArea)

# Get the bounding box of the largest contour
x, y, w, h = cv.boundingRect(largest_contour)

# Dynamically exclude the bottom part of the bounding box
barcode_only_height = int(h * 0.85)  # Retain only 85% of the height
cropped_img = img[y:y+barcode_only_height, x:x+w]
cv.imshow('Cropped Barcode', cropped_img)

# Draw contours on the original image
contour_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert binary image to BGR
cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

# Resize the image to increase its dimensions
scale_factor = 4  # Adjust the scale factor as needed
resized_img = cv.resize(cropped_img, (cropped_img.shape[1] * scale_factor, cropped_img.shape[0] * scale_factor))
cv.imshow('Resized Image', resized_img)



# Binarize the image using a threshold
_, binary_img = cv.threshold(resized_img, 128, 255, cv.THRESH_BINARY_INV)
cv.imshow('Binarized Image', binary_img)
# Create a custom structuring element for closing
kernel_height1 = 9  # Adjust the height as needed
kernel_width1 = 9  # Adjust the width as needed
kernel1 = np.zeros((kernel_height1, kernel_width1), np.uint8)
kernel1[:, kernel_width1 // 2] = 1  # Set the middle column to 1's

# Apply morphological opening to close gaps in the barcode
# kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size based on gap size
binary_img1 = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel1)
cv.imshow('Morphologically Opened Image', binary_img1)


# Create a custom structuring element for closing
kernel_height1 = 17  # Adjust the height as needed
kernel_width1 = 17  # Adjust the width as needed
kernel1 = np.zeros((kernel_height1, kernel_width1), np.uint8)
kernel1[:, kernel_width1 // 2] = 1  # Set the middle column to 1's
# Apply morphological closing to close gaps in the barcode
binary_img = cv.morphologyEx(binary_img1, cv.MORPH_CLOSE, kernel1)
cv.imshow('Morphologically Closed Image', binary_img)

# Create a custom structuring element for horizontal dilation
kernel_height2 = 1  # Height of 1 for horizontal line
kernel_width2 = 7  # Adjust the width as needed
kernel2 = np.ones((kernel_height2, kernel_width2), np.uint8)  # Horizontal line of 1's

# Apply morphological closing to close gaps in the barcode
binary_img = cv.morphologyEx(binary_img1, cv.MORPH_CLOSE, kernel2)
cv.imshow('Morphologically Closed Image', binary_img)

# Invert the morphologically closed image
inverted_img = cv.bitwise_not(binary_img)
cv.imshow('Inverted Morphologically Closed Image', inverted_img)



cv.waitKey(0)
cv.destroyAllWindows()