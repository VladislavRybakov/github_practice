import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate
from skimage.transform import rescale


def radon_transform(PATH):
    img = cv2.imread(PATH, 0)  # read image in grayscale
    # img = rescale(img, scale=1.4)
    sinogram = radon(img, theta=np.arange(89., 90.))

    # plt.imshow(sinogram, cmap=plt.cm.Greys_r, aspect='auto')

    # plt.plot(sinogram)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax1.imshow(img, cmap=plt.cm.Greys_r)
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r, aspect='auto')

    plt.show()


def discrete_radon_transform(image):
    steps = len(sum(rotate(image, 0)))
    R = np.zeros((len(image), steps), dtype='float64')
    image = np.transpose(image)
    for s in range(steps):
        #rotation = rotate(image, -s*180/steps)
        #rotation = rotate(image, 90)

        rotation = rotate(image, 0)
        R[:,s] = sum(rotation)
    return R


PATH = 'C:\\Users\\rybak\\Desktop\\Python programms\\MyProject\\mute_gray.png'
# Read image as 64bit float gray scale
image = cv2.imread(PATH, 0)
radon = discrete_radon_transform(image)

# Plot the original and the radon transformed image
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(radon, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()