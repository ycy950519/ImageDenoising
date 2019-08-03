#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dataset
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D,LeakyReLU
from skimage.measure import compare_ssim as ssim

batch_size=30
#输入测试图像
testimages=dataset.show_images('Image_Denoising/',256)
#lrelu函数
def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

#输入到网络的数据
#图片28*28,color_channel是1
img_input = tf.placeholder(tf.float32, [None, 256, 256, 1])
#原始数据
targets_ = tf.placeholder(tf.float32, [None, 256, 256, 1])
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(img_input, filters=32,kernel_size=(5, 5),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu )
conv1=tf.layers.conv2d(conv1,filters=64,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,activation=tf.nn.relu)

conv2 = tf.layers.conv2d(img_input, filters=64,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu )
conv3 = tf.layers.conv2d(img_input, filters=64,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu)

avepool=tf.layers.average_pooling2d(img_input,pool_size=(3,3),strides=1,padding='SAME')
avepool=tf.layers.conv2d(avepool,filters=32,kernel_size=(1,1),strides=(1,1),padding='SAME',use_bias=True,activation=tf.nn.relu)

conv_1=tf.concat(axis=3,values=[conv1,conv2,conv3,avepool])
conv_1=tf.layers.batch_normalization(conv_1,axis=3)

maxpool1 = tf.layers.max_pooling2d(conv_1,pool_size=(2, 2),strides=(2, 2) )


conv4 = tf.layers.conv2d(maxpool1, filters=32,kernel_size=(5,5),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv5 = tf.layers.conv2d(maxpool1, filters=32,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv6 = tf.layers.conv2d(maxpool1, filters=32,kernel_size=(9,9),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv6_6=tf.layers.conv2d(maxpool1, filters=32,kernel_size=(7,7),strides=(1,1),padding='SAME',use_bias=True,activation='relu')
conv_2=tf.concat(axis=3,values=[conv5,conv4,conv6_6,conv6])
conv_2=tf.layers.batch_normalization(conv_2,axis=3)



upsamples2 = tf.layers.conv2d_transpose(conv_2,filters=32,kernel_size=(3,3),padding='SAME',strides=(2,2),name='upsamples2')
upsamples2=tf.layers.dropout(upsamples2)
print(upsamples2.shape)


logits = tf.layers.conv2d_transpose(upsamples2,filters=1,kernel_size=(3, 3),strides=(1, 1),name='logits',padding='SAME',use_bias=True)
logits=tf.layers.batch_normalization(logits,axis=3)

# 此时的数据是 256x256x1
    # 通过sigmoid传递logits以获得重建图像

logits_=tf.add(img_input,logits)
logits_=tf.nn.relu(logits_)

decoded = tf.sigmoid(logits_, name='recon')

with tf.name_scope('cost'):
    # 定义损失函数和优化器
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=targets_)
    #误差
    cost = tf.reduce_mean(loss)
    tf.summary.scalar('cost', cost)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#设置要保存的变量
#设置要保存的变量
saver = tf.train.Saver(tf.global_variables())
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('logs/test_logs',sess.graph_def)
    #恢复变量
    saver.restore(sess,'encode_model1/model_cnn_train27_best.ckpt')
    index=0
    #添加噪声
    #noisy_im = sess.run(noisy_image,feed_dict={content_image_place_shape:content_image.shape,content_image_place:content_image/1.0})
    for i in testimages:
        index+=1
        img = i
        noisyImage = dataset.gaussian(i, 25)
        # img = dataset.pretrain_images(img)
        # noisyImage = dataset.pretrain_images(noisyImage)
        if img.ndim==2:
            img=img[None,...,None]
        if noisyImage.ndim==2:
            noisyImage=noisyImage[None,...,None]
        img=dataset.pretrain_images(img)
        noisyImage=dataset.pretrain_images(noisyImage)

        #预测
        pred = sess.run(decoded, feed_dict={img_input:noisyImage})

        psnr_score1 = tf.image.psnr(pred, img, max_val=1)  # 去噪图psnr
        psnr_score2 = tf.image.psnr(noisyImage, img, max_val=1)  # 噪声图psnr
        # ssim1=ssim(pred, img,multichannel=True)

        with tf.Session() as sess1:
            psnr_score1 = sess1.run(psnr_score1)
            psnr_score2 = sess1.run(psnr_score2)

        print(psnr_score1)
        # print(ssim1)

        im_out_denoisy = np.clip(pred[0, ..., 0] * 255, 0, 255).astype(np.uint8)
        # im_noisy = np.clip(img[0, ...,0]*255,0,255).astype(np.uint8)
        cv2.imwrite('Image_test\\' + str(index) + '.jpg', im_out_denoisy)
        # cv2.imwrite('Image_test_Noise\\'+str(index)+'.jpg',)


# im_out_noisy = np.clip(test_x[0, ...,0]*255,0,255).astype(np.uint8)
# im_out_denoisy = np.clip(pred[0, ...,0]*255,0,255).astype(np.uint8)

#Clear Image


#显示图像
# plt.subplot(131)
# plt.title('noisy')
# plt.imshow(img_normal,cmap='gray')
#Noisy Image
plt.subplot(132)
plt.title('noisy')
# plt.imshow(im_noisy,cmap='gray')
#Denoisy Image
# plt.subplot(133)
plt.title('pred')
plt.imshow(im_out_denoisy,cmap='gray')
plt.show()
