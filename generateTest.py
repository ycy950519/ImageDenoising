#coding=utf-8
import cv2
import h5py
import numpy as np
import random
import os
import glob

ROOT_VOC_DIRECTORY = ''


#读取所有图片的名字
def readImageName():
    namesFile=ROOT_VOC_DIRECTORY+'Image_Denoising/'
    allImage = []
    # fd = open(namesFile, "r")
    path = os.path.join(namesFile, '*g')
    files = glob.glob(path)
    for fl in files:
        image = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)
        allImage.append(image)
    # fd.close()
    return allImage

#添加椒盐噪声
def SaltAndPepper(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg
#添加高斯噪声
def GaessNoisy(src,sigma):
    NoiseImg = src.copy()
    s = np.random.normal(0, 1, size=src.shape)*sigma
    NoiseImg = np.add(NoiseImg,s)
    NoiseImg.astype(dtype=np.uint8)
    # cv2.imwrite(ROOT_VOC_DIRECTORY + 'ImageNoise//' + str() + '.jpeg', NoiseImg)  # 将噪声图片保存
    return NoiseImg


def readImage2Data(pathcImageNames,patchSize,stride):
    patchImageInput = np.empty(shape=[0,patchSize,patchSize,1])
    patchImageLabel = np.empty(shape=[0,patchSize,patchSize,1])
    for i in pathcImageNames:
        # print(len(pathcImageNames))
        # img = cv2.imread(ROOT_VOC_DIRECTORY+'Image/'+pathcImageNmaes[i],0)
        img=i
        #添加椒盐噪声
        #noisyImage = SaltAndPepper(img,0.2)
        #添加高斯噪声
        noisyImage = GaessNoisy(img,25)

        #re to 0-1

        img = img/255.0
        noisyImage = noisyImage/255.0
        # cv2.imshow('i',img)
        # cv2.imshow('img',noisyImage)
        # cv2.waitKey()
        row = (img.shape[0]-patchSize)//stride
        line = (img.shape[1]-patchSize)//stride
        imgPatch = np.zeros(shape=[row*line,patchSize,patchSize,1])
        imgPatchLabel = np.zeros(shape=[row*line,patchSize,patchSize,1])
        for r in range(row):
            for l in range(line):
                imgPatch[r*line+l,...,0]=img[r*stride:r*stride+patchSize,l*stride:l*stride+patchSize]
                imgPatchLabel[r*line+l,...,0]=noisyImage[r*stride:r*stride+patchSize,l*stride:l*stride+patchSize]

        patchImageInput = np.vstack((patchImageInput,imgPatch))
        patchImageLabel = np.vstack((patchImageLabel,imgPatchLabel))
        print(patchImageInput)

    return patchImageInput,patchImageLabel



def writeData2H5(data,label,batchSize,fileNum):
    file = h5py.File(ROOT_VOC_DIRECTORY+'testimage/test{0}.h5'.format(fileNum),'w')
    data = data[0:(data.shape[0]-data.shape[0]%batchSize),...]
    label = label[0:(label.shape[0]-label.shape[0]%batchSize),...]
    #random
    '''
    randomM = list(xrange(data.shape[0]))
    random.shuffle(randomM)
    data = data[randomM,...]
    label = label[randomM,...]
    '''
    carh5data = file.create_dataset('data',data=data,shape=data.shape,dtype=np.float32)
    carh5label = file.create_dataset('label',data=label,shape=label.shape,dtype=np.float32)
    file.close()


if __name__ == '__main__':
    allImageNames = readImageName()
    PATCH_SIZE = 20     #片段大小
    STRIDE_SIZE = 30    #步长大小
    BATCH_SIZE = 128    #训练一个批次的大小
    NUM_ONE_H5 = 1   #一个h5文件放图片的数量
    NUM_H5 = 1        #h5文件数量
    NUM_BEGIN = 0     #第几个h5文件开始
    NUM_H5_MAX = len(allImageNames)//NUM_ONE_H5
    for i in range(NUM_BEGIN,NUM_H5_MAX):
        # print(i*NUM_ONE_H5)
        patchImageName = allImageNames[i*NUM_ONE_H5:(i+1)*NUM_ONE_H5]
        data,label = readImage2Data(patchImageName,PATCH_SIZE,STRIDE_SIZE)
        writeData2H5(data,label,BATCH_SIZE,i)



