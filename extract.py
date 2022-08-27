from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, rotate
import statistics

# извлечение НЦВЗ

def main():
    PATH = "C:\\Users\\rybak\\Desktop\\Python programms\\MyProject\\picture_test_d2_hse.jpg"

    gray_im = cv2.imread(PATH, 0) # считывание изображения в оттенках серого
    #plt.imshow(gray_im, cmap='gray') #для проверки что изображение реально в серых тонах
    #plt.show()
    pixels = gray_im
    width, heigth = len(gray_im[0]), len(gray_im) # длина и ширина изображения

    def mute_im(n): # среднеквадратичная фильтрация, где квадрат размером n*n
        pix = np.zeros((heigth, width), dtype='float64')
        for i in range(heigth - 1): # алгоритм ставит каждому пикселю среднее
                                    # арифметическое значение всех пикселей внутри квадрата размером n*n
            for j in range(width - 1):
                if i < n:
                    first_i = 0
                else:
                    first_i = i - n

                if j < n:
                    first_j = 0
                else:
                    first_j = j - n

                if j + n >= width:
                    last_j = width - 1
                else:
                    last_j = j + n

                if i + n >= heigth:
                    last_i = heigth - 1
                else:
                    last_i = i + n

                sred = 0

                for a in range(first_i, last_i + 1):
                    for b in range(first_j, last_j + 1):
                        sred += pixels[a][b]

                '''
                a = slice(first_i, last_i + 1, 1)
                b = slice(first_j, last_j + 1, 1)
                srez = pixels[a][b]
                print(srez)
                print(first_i,  last_i + 1, first_j, last_j + 1)
                sred = np.average(srez)
                '''
                sred -= pixels[i][j]
                sred = sred // ((last_i - first_i + 1) * (last_j - first_j + 1) - 1)
                pix[i][j] = sred

        '''
        data = []
        for i in range(len(pix)):
            for j in range(len(pix[i])):
                data.insert(len(data), pix[i][j])

        mute_img = Image.new("L", (width, heigth), 255) # для сохранения изображения, если надо
        mute_img.putdata(data)
        mute_img.save('C:\\Users\\rybak\\Desktop\\Python programms\\MyProject\\mute_gray.png')
        '''

        return pix

    muted_image = mute_im(9)


    def discrete_radon_transform(image): # дискретное преобразование радона, возвращает
        R = np.zeros(len(image), dtype='float64')
        image = np.transpose(image)
        rotation = rotate(image, 0)
        R[:] = sum(rotation)
        return R

    sinogram = discrete_radon_transform(muted_image)
    sinogram = sinogram[:-1]

    strings = []
    strings_points = []
    max = sorted(sinogram)
    max = max[len(max)-1] * 0.98

    for i in range(len(sinogram)): # находит координаты строк и значение функции радона в этой строке
        if i > 0 and i < len(sinogram)-1:
            if sinogram[i - 1] > sinogram[i] and sinogram[i] < sinogram[i + 1] and sinogram[i] < max:
                strings.insert(len(strings), i)
                strings_points.insert(len(strings_points), sinogram[i])

    print("Strings number: ", len(strings))
    print(strings)
    print(strings_points)

    intervals = []
    for i in range(len(strings)): # создание массива со значениями длины интервалов
        if i != 0:
            intervals.insert(len(intervals), strings[i] - strings[i-1])


    print("Межстрочные интервалы: ", intervals)
    mean = statistics.mean(intervals)
    print(mean)
    i = 0
    while i < len(intervals) - 1: #удаление ошибок
        if intervals[i] < mean * 0.8:
            intervals[i] += intervals[i+1]
            intervals.pop(i+1)
        if intervals[i] > mean * 1.5:
            intervals.pop(i)
            continue
        i += 1

    i = 0
    while i < len(intervals) - 1:  # удаление ненужного хвоста, где куча фейковых строк
        if intervals[i] < mean * 0.8:
            del intervals[i:]
            break
        if intervals[i] > mean * 1.5:
            intervals.pop(i)
            continue
        i += 1

    print("Strings number: ", len(intervals) + 1)

    print("Межстрочные интервалы: ", intervals)

    mean = statistics.mean(intervals)
    std = statistics.stdev(intervals)
    print("Std: ", statistics.stdev(intervals))
    print("Mean: ", statistics.mean(intervals))

    if std < 0.4: # проверка есть ли вообще в документе искажения
        print("Водяной знак отсутствует")
        #return 0

    whatermark = ""
    for i in intervals:
        if i > mean + std * 0.7:
            whatermark += "1"
        else:
            whatermark += "0"

    print("Whatermark: \t\t", whatermark)
    print("Original whatermark:\t 010010000101001101000101")
    #plt.imshow(sinogram, cmap=plt.cm.Greys_r, aspect='auto')
    plt.plot(sinogram)
    plt.show()
    return whatermark
    #Whatermark =  010010000101001101000101

if __name__ == "__main__":
    main()