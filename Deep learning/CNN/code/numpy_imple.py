import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

def conv2d(image, kernel, padding, stride):
    image = np.pad(image, pad_width= padding, mode= 'constant')
    r_image, c_image = image.shape
    r_kernel, c_kernel = kernel.shape
    out_height = (r_image - r_kernel) // stride + 1
    out_width = (c_image - c_kernel) // stride + 1
    res = np.zeros((out_height, out_width))
    for i in range (out_height):
        for j in range (out_width):
            tmp = image[i * stride:i * stride + r_kernel, j * stride : j * stride + c_kernel]
            res[i, j] = (tmp * kernel).sum()
    return res
