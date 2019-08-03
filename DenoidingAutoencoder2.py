import dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import cv2
from skimage.measure import compare_ssim as ssim

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset.remove_files('./graphs1')
dataset.remove_files('./encode_model1')
# dataset.remove_files('./Image_Gaussian_Denoising')
# dataset.remove_files('./Image_Salt_and_Pepper_Denoising')



print('开始数据预处理...')

#读取并展示未加噪声的图片
path_images_without_noise = 'Image_Denoising'
batch_size=128
file_num=70
epoch=100
IMAGE_WIDTH=20
step=0
max_psnr=0

def loadH5data(filenum):
    file = h5py.File('h5/train{0}.h5'.format(filenum),'r')
    data_train = file['data'][:]
    data_label = file['label'][:]
    hw_train = data_train.shape[2]
    hw_label = data_label.shape[2]
    data_train = data_train.reshape((data_train.shape[0],hw_train,hw_train,1))
    data_label = data_label.reshape((data_label.shape[0],hw_label,hw_label,1))
    return data_train,data_label

def psnr_img():
     # 所有文件全路径列表
    image = cv2.imread('testimage/lena1.jpg')  # 通过路径把图读进来
    image = cv2.resize(image, (20, 20), 0, 0, cv2.INTER_LINEAR)  # 图片重塑,shape是(256, 256)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noise=dataset.gaussian(image,25)
    if image.ndim==2:
        image=image[None,...,None]
    if noise.ndim == 2:
        noise = noise[None, ..., None]
    noise = dataset.pretrain_images(noise)
    image = dataset.pretrain_images(image)
    return image,noise

print('开始训练...')

#lrelu函数
def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

#输入到网络的数据
#图片28*28,color_channel是1
inputs_1 = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_WIDTH, 1])
#原始数据
targets_ = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_WIDTH, 1])
learning_rate = tf.placeholder(tf.float32)


### Encoder
'''
filters: 32个卷积核
kernel_size: 卷积核大小
strides: 长宽步长都是1
padding: 边缘填充
use_bias: 在卷积中添加偏置
activation: 激活函数
'''
#
conv1 = tf.layers.conv2d(inputs_1, filters=32,kernel_size=(5, 5),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu )
conv1=tf.layers.conv2d(conv1,filters=64,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,activation=tf.nn.relu)

conv2 = tf.layers.conv2d(inputs_1, filters=64,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu )
conv3 = tf.layers.conv2d(inputs_1, filters=64,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation=tf.nn.relu)

avepool=tf.layers.average_pooling2d(inputs_1,pool_size=(3,3),strides=1,padding='SAME')
avepool=tf.layers.conv2d(avepool,filters=32,kernel_size=(1,1),strides=(1,1),padding='SAME',use_bias=True,activation=tf.nn.relu)

conv_1=tf.concat(axis=3,values=[conv1,conv2,conv3,avepool])
conv_1=tf.layers.batch_normalization(conv_1,axis=3)
conv_1=tf.layers.dropout(conv_1)

# conv4 = tf.layers.conv2d(inputs_1, filters=32,kernel_size=(5,5),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
# conv5 = tf.layers.conv2d(conv_1, filters=64,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
# conv6 = tf.layers.conv2d(conv5, filters=64,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
# conv6_6=tf.layers.conv2d(maxpool1, filters=32,kernel_size=(7,7),strides=(1,1),padding='SAME',use_bias=True,activation='relu')
# conv_2=tf.concat(axis=3,values=[conv5,conv4,conv6_6,conv6])
# conv_2=tf.layers.batch_normalization(conv_2,axis=3)

conv4 = tf.layers.conv2d(conv_1, filters=64,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )

conv5=tf.layers.conv2d(conv_1, filters=48,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv6=tf.layers.conv2d(conv5, filters=64,kernel_size=(5,5),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )

conv7=tf.layers.conv2d(conv_1, filters=64,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv8=tf.layers.conv2d(conv7, filters=96,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv9=tf.layers.conv2d(conv8, filters=96,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )

avepool1=tf.layers.average_pooling2d(conv_1,pool_size=(3,3),strides=1,padding='SAME')
avepool1=tf.layers.conv2d(avepool1,filters=32,kernel_size=(1,1),strides=(1,1),padding='SAME',use_bias=True,activation=tf.nn.relu)

conv_2=tf.concat(axis=3,values=[conv4,conv6,conv9,avepool1])
conv_2=tf.layers.batch_normalization(conv_2,axis=3)
conv_2=tf.layers.dropout(conv_2)



# upsamples2 = tf.layers.conv2d_transpose(conv6,filters=32,kernel_size=(3,3),padding='SAME',strides=(1,1),name='upsamples2')
# upsamples2=tf.layers.dropout(upsamples2)
# print(upsamples2.shape)

conv7 = tf.layers.conv2d_transpose(conv_2, filters=16,kernel_size=(5,5),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv8 = tf.layers.conv2d_transpose(conv_2, filters=16,kernel_size=(3,3),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
conv9 = tf.layers.conv2d_transpose(conv_2, filters=16,kernel_size=(1,1),strides=(1, 1),padding='SAME',use_bias=True,activation='relu', )
# conv10=tf.layers.conv2d_transpose(conv_2, filters=16,kernel_size=(7,7),strides=(2,2),padding='SAME',use_bias=True,activation='relu')
conv_3=tf.concat(axis=3,values=[conv7,conv8,conv9])
conv_3=tf.layers.batch_normalization(conv_3,axis=3)
conv_3=tf.layers.dropout(conv_3)
# print(conv_3.shape)


logits = tf.layers.conv2d_transpose(conv_3,filters=1,kernel_size=(3, 3),strides=(1, 1),name='logits',padding='SAME')
# logits=tf.layers.batch_normalization(logits,axis=3)

# 此时的数据是 256x256x1
    # 通过sigmoid传递logits以获得重建图像
#
# logits_=tf.add(inputs_1,logits)
# logits_=tf.nn.relu(logits_)

decoded = tf.sigmoid(logits, name='recon')

    # 定义损失函数和优化器
loss = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits, labels=targets_)
#误差
cost = tf.reduce_mean(loss)
tf.summary.scalar('cost', cost)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#class_name噪声种类
#graph_name sess的默认图graph存在哪个文件夹
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    lr = 1e-3
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs1', sess.graph)  # 再graph文件夹下使用默认图

    for e in range(500):  # 每一个epoch
        for i in range(file_num):
            datatrain, datanoise = loadH5data(i)
            data_batch_num = datatrain.shape[0] // batch_size
            for batch in range(data_batch_num):
                temp_input = datatrain[batch * batch_size:(batch + 1) * batch_size,(55-IMAGE_WIDTH)//2:int(55-(55-IMAGE_WIDTH)/2),(55-IMAGE_WIDTH)//2:int(55-(55-IMAGE_WIDTH)/2),...]
                temp_label = datanoise[batch * batch_size:(batch + 1) * batch_size,(55-IMAGE_WIDTH)//2:int(55-(55-IMAGE_WIDTH)/2),(55-IMAGE_WIDTH)//2:int(55-(55-IMAGE_WIDTH)/2),...]
                summary, train_loss, _ = sess.run([merged, cost, opt],
                                                  feed_dict={inputs_1: temp_label, targets_: temp_input,
                                                             learning_rate: lr * (0.997) ** e})
                print('第Epoch : ', e, '批  Training Cost : ', train_loss, ' Learning Rate : ', lr * (0.997) ** e)

                step+=1
                if step%10==0:
                    psnr_image,psnr_noise=psnr_img()
                    reconstructed = sess.run(decoded, feed_dict={inputs_1: psnr_noise})



                    psnr_score1 = tf.image.psnr(reconstructed, psnr_image, max_val=1)  # 去噪图psnr
                    psnr_score2 = tf.image.psnr(psnr_noise, psnr_image, max_val=1)  # 噪声图psnr


                    with tf.Session() as sess1:
                        psnr_score1 = sess1.run(psnr_score1)
                        psnr_score2 = sess1.run(psnr_score2)

                    # print(psnr_score2,'  ',ssmi1,'  ',psnr_score1,'  ',ssmi2)
                    print(psnr_score1,'  ',psnr_score2)
                    with open('result1.txt','a') as f:
                        f.write(str(e)+'  '+str(psnr_score1)+'  '+str(psnr_score2))
                        f.write('\r\n')
                    if psnr_score1>max_psnr:
                        max_psnr=psnr_score1
                        saver.save(sess, 'encode_model1/model_cnn_train{0}_best.ckpt'.format(e))

            if e%5==0:
                saver.save(sess, 'encode_model1/model_cnn_train{0}.ckpt'.format(e))
                writer.add_summary(summary, e)



print('Finish')