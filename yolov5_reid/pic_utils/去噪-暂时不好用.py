import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
def quzao(img_path):
    img = cv.imread(img_path)
    dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.savefig("light__.jpg")

quzao('light_.jpg')