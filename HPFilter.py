import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

def plot(data, title):
    plot.i += 1
    plt.subplot(3, 2, plot.i)
    plt.imshow(data, cmap='gray')
    plt.title(title)

plot.i = 0

# Loading data
img = Image.open('testImage1.png')
data = np.array(img, dtype=float)

# Convert image to grayscale
if len(data.shape) == 3:
    data = data.mean(axis=2)
plot(data, 'Grayscale Image')

# Apply Gaussian smoothing for denoising
smoothed = ndimage.gaussian_filter(data, 3)
plot(smoothed, 'Smoothed Image')

# Apply highpass filter
# Middle value is higher for more contrast, where flame should be
# Kernel values may need adjusted
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
highpass = ndimage.convolve(smoothed, kernel)
plot(highpass, 'Highpass Filter')

# Binary mask to completely filter out background
# Threshold values may need adjusted
threshold = np.mean(highpass) * 1.5 # Adjustable weight to show flame
flame_highlighted = data.copy()
flame_highlighted[highpass <= threshold] = 0
plot(flame_highlighted, 'Flame Highlighted')

plt.tight_layout()
plt.show()
plt.savefig('filter.png')
