import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate, radon
from skimage.transform import rescale


def radon_transform(image): # https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html
    theta = np.linspace(0., 180., min(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram # просто получение синограммы с помощью библиотечной функции


def discrete_radon_transform(image):
    steps = len(sum(rotate(image, 0))) # я не знаю что это и зачем
    sinogram = np.zeros((len(image), steps), dtype='float64')
    image = np.transpose(image)  # финт ушами чтобы все работало
    # rotation = rotate(image, 0)
    for s in range(steps):
        rotation = rotate(image, s*180/steps + 90) # вращает изображение и возвращает что-то
        sinogram[:,s] = sum(rotation) # я тоже не знаю что это...
    return sinogram


PATH = 'C:\\Users\\rybak\\Desktop\\Python programms\\MyProject\\mute_gray.png'

image = cv2.imread(PATH, 0)
discrete_sinogram = discrete_radon_transform(image)
sinogram = radon_transform(image)

# Plot the original and the radon transformed image
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(discrete_sinogram, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(sinogram, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 1), plt.title("Original picture")
plt.subplot(1, 3, 2), plt.title("Discrete radon transformation")
plt.subplot(1, 3, 3), plt.title("Radon transformation")
plt.show()