import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

# Loading data
img = Image.open('testImage1.png')
data = np.array(img, dtype=float)
