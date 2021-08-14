import numpy as np

import cv2
# 加载图像
def light(img_path):
    image = cv2.imread(img_path)
    (B, G, R) = cv2.split(image)
    imageBlueChannelAvg = np.mean(B)
    imageGreenChannelAvg = np.mean(G)
    imageRedChannelAvg = np.mean(R)
    K = (imageRedChannelAvg+imageGreenChannelAvg+imageRedChannelAvg)/3;
    Kb = K/imageBlueChannelAvg;
    Kg = K/imageGreenChannelAvg;
    Kr = K/imageRedChannelAvg
    B = cv2.addWeighted(B, Kb, 0, 0, 0)
    G = cv2.addWeighted(G, Kg, 0, 0, 0)
    R = cv2.addWeighted(R, Kr, 0, 0, 0)
    image_new = cv2.merge([B, G, R])

#     # cv2.imshow("des", image_new)
#     cv2.imwrite('light_.jpg', image_new)
# light('light.jpg')


#全局直方图均衡化
def eaualHist_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    #opencv的直方图均衡化要基于单通道灰度图像
#     cv.namedWindow('input_image', cv.WINDOW_NORMAL)
#     cv.imshow('input_image', gray)
    dst = cv2.equalizeHist(gray)                #自动调整图像对比度，把图像变得更清晰
    cv2.namedWindow("eaualHist_demo", cv2.WINDOW_NORMAL)
    cv2.imshow("eaualHist_demo", dst)
    return dst


def clahe_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(5, (8,8))
    dst = clahe.apply(gray)
    cv2.imwrite('light_.jpg',dst)
    # cv2.namedWindow("clahe_demo", cv2.WINDOW_NORMAL)
    # cv2.imshow("clahe_demo", dst)

# src=cv2.imread('light.jpg')
# clahe_demo((src))
# # dst = eaualHist_demo(src)
# # dst = clahe_demo(src)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
