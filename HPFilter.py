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
