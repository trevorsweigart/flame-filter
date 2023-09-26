import cv2
import numpy as np

# Load the image
image = cv2.imread('fire.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Threshold the image to create a binary mask
_, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the detected contours on the original image
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 0, 255), 2)  # Red color for contours

# Display the result
cv2.imshow('Flame Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
