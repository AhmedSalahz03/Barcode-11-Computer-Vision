import cv2 as cv
import numpy as np
import random

def add_salt_and_pepper_noise(image, amount=0.1):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Add salt
    for _ in range(num_salt):
        y, x = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[y, x] = 255

    # Add pepper
    for _ in range(num_pepper):
        y, x = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[y, x] = 0

    return noisy_image

# Load the image
image = cv.imread('Samples/Test Cases/01 - lol easy.jpg', cv.IMREAD_GRAYSCALE)
noisy_image = add_salt_and_pepper_noise(image, amount=0.05)

# Display
cv.imshow('Original Image', image)
cv.imshow('Salt and Pepper Noise', noisy_image)


def add_scratching_effect(image, num_scratches=10, thickness_range=(1, 2), color=255):
    """
    Adds random scratches to an image.
    
    Parameters:
        image (numpy.ndarray): Input grayscale image.
        num_scratches (int): Number of scratches to add.
        thickness_range (tuple): Range of thickness for scratches (min, max).
        color (int): Color of scratches (default is 255 for white scratches).
        
    Returns:
        numpy.ndarray: Image with scratching effect.
    """
    scratched_image = image.copy()
    rows, cols = image.shape
    
    for _ in range(num_scratches):
        # Random start and end points for a scratch
        x1, y1 = random.randint(0, cols - 1), random.randint(0, rows - 1)
        x2, y2 = random.randint(0, cols - 1), random.randint(0, rows - 1)
        
        # Random thickness for the scratch
        thickness = random.randint(thickness_range[0], thickness_range[1])
        
        # Draw the scratch on the image
        cv.line(scratched_image, (x1, y1), (x2, y2), color, thickness)
    
    return scratched_image

# Load the image
image = cv.imread('Samples/Test Cases/01 - lol easy.jpg', cv.IMREAD_GRAYSCALE)

# Add scratching effect
scratched_image = add_scratching_effect(image, num_scratches=20, thickness_range=(1, 3), color=255)

# Display the original and scratched images
cv.imshow('Original Image', image)
cv.imshow('Scratched Image', scratched_image)

def blur_image(image, kernel_size=(5, 5), sigma=0):
    """
    Blurs an image using a Gaussian filter.
    
    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (tuple): Kernel size for the Gaussian filter.
        sigma (float): Standard deviation for Gaussian filter.
        
    Returns:
        numpy.ndarray: Blurred image.
    """
    return cv.GaussianBlur(image, kernel_size, sigma)

blurred_image = blur_image(image, kernel_size=(13, 13), sigma=0)

# Display the original and blurred images
cv.imshow('Original Image', image)
cv.imshow('Blurred Image', blurred_image)

def add_sinusoidal_noise(image, frequency=10, amplitude=50, direction='horizontal'):
    """
    Adds sinusoidal wave noise to an image.
    
    Parameters:
        image (numpy.ndarray): Input grayscale image.
        frequency (int): Frequency of the sine wave (number of cycles per image width/height).
        amplitude (int): Amplitude of the sine wave (strength of the noise).
        direction (str): Direction of the wave ('horizontal', 'vertical', or 'both').
        
    Returns:
        numpy.ndarray: Image with sinusoidal noise.
    """
    rows, cols = image.shape
    noisy_image = image.copy().astype(np.float32)

    if direction in ('horizontal', 'both'):
        # Create horizontal sine wave
        x = np.arange(cols)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x / cols)
        for y in range(rows):
            noisy_image[y, :] += sine_wave

    if direction in ('vertical', 'both'):
        # Create vertical sine wave
        y = np.arange(rows)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * y / rows)
        for x in range(cols):
            noisy_image[:, x] += sine_wave

    # Clip the values to the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# Add sinusoidal noise
noisy_image = add_sinusoidal_noise(image, frequency=20, amplitude=90, direction='horizontal')

# Display the images
cv.imshow('Original Image', image)
cv.imshow('Sinusoidal Noise', noisy_image)

cv.waitKey(0)
cv.destroyAllWindows()



