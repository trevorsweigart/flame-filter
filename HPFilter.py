import matplotlib
import numpy as np
from scipy import ndimage
from scipy.ndimage import center_of_mass, convolve, gaussian_filter
from PIL import Image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Loading data
img = Image.open('Image1.jpg')
data = np.array(img, dtype=float)

# Convert image to grayscale
if len(data.shape) == 3:
    data = data.mean(axis=2)

# Apply Gaussian smoothing for denoising
smoothed = ndimage.gaussian_filter(data, 3)

# Apply highpass filter
# Middle value is higher for more contrast, where flame should be
# Kernel values may need adjusted
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
highpass = ndimage.convolve(smoothed, kernel)

# Binary mask to completely filter out background
# Threshold values may need adjusted
threshold = np.mean(highpass) * 1.7 # Adjustable weight to show flame
flame_highlighted = data.copy()
flame_highlighted[highpass <= threshold] = 0

# Calculate the centroid of the binary mask
centroid = center_of_mass(flame_highlighted)
image_center = np.array(flame_highlighted.shape) / 2

fig, ax = plt.subplots(figsize=(data.shape[1] / 100, data.shape[0] / 100))

# Display the centroid as a red dot on the final image
ax.imshow(flame_highlighted, cmap='gray')
ax.scatter(centroid[1], centroid[0], c='red', marker='x', s=100, label='Centroid')

# Draw the acceptable centered area
center_acceptance_radius = 0.2 * min(flame_highlighted.shape)
acceptable_center_area = patches.Circle(image_center[::-1], center_acceptance_radius, color='blue', fill=False, linestyle='--', linewidth=1.5, label='Acceptable Center Area')
ax.add_patch(acceptable_center_area)

ax.set_title('Masked flame with centroid and centering acceptance')
ax.legend()
ax.axis('off')

flame_area = np.sum(flame_highlighted > 0)

# Minimum acceptable area
min_area = 500  # Adjustable threshold

# Check if heat source is big enough
if flame_area >= min_area:
    print(f"Flame is big enough with area: {flame_area}")
    flame_big_enough = True
else:
    print(f"Flame is too small with area: {flame_area}")
    flame_big_enough = False

# Check if the centroid is centered
distance_to_center = np.linalg.norm(np.array(centroid) - image_center)
if distance_to_center < 0.2 * min(flame_highlighted.shape):
    print("Flame is centered.")
else:
    print("Flame is not centered.")

plt.savefig('filtered_image.jpg', bbox_inches='tight', pad_inches=0)

