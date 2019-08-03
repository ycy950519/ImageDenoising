import cv2
import numpy as np
import dataset
import tensorflow as tf

# img01 = cv2.imread("E:\cpy\photo\image01.bmp")  # 读取目标图片
testdata=dataset.show_images('Image_Denoising',256)
# 中值滤波
for i in testdata:
    img=i
    img_noise=dataset.gaussian(img,25)

    img=dataset.pretrain_images(img)
    img_noise=dataset.pretrain_images(img_noise)

    img_medianBlur = cv2.medianBlur(img_noise, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 均值滤波
    img_Blur = cv2.blur(img_noise, (5, 5))
    img_Blur=dataset.pretrain_images(img_Blur)
    # 高斯滤波
    img_GaussianBlur = cv2.GaussianBlur(img_noise, (7, 7), 0)
    img_GaussianBlur=dataset.pretrain_images(img_GaussianBlur)

    # 高斯双边滤波
    img_bilateralFilter = cv2.bilateralFilter(img_noise, 40, 75, 75)
    img_bilateralFilter=dataset.pretrain_images(img_bilateralFilter)

    psnr1=tf.image.psnr(img_medianBlur,img,max_val=1)
    psnr2=tf.image.psnr(img_GaussianBlur,img,max_val=1)
    psnr3=tf.image.psnr(img_bilateralFilter,img,max_val=1)
    print(psnr1,psnr2,psnr3)