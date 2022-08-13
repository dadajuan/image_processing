#from dataset import  TrafficLightDataset
from PIL import Image
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import time
import pandas as pd



dataset_dir = 'D:/浏览器下载/PTL_Dataset_4032x3024_Part1'
savedataset_dir = 'D:/浏览器下载/PTL_Dataset_4032x3024_Part5'
number_dir = 'D:/浏览器下载/PIL_dataset_P1_light_test_504_T2'
csv_file_2 = 'D:/浏览器下载/Annotations/training_file2.csv'
csv_file_3 = 'D:/浏览器下载/Annotations/newnew.csv'
# with open(csv_file_2, 'r') as f:
#     m= len(f.readlines())
#     print(m)
labels = pd.read_csv(csv_file_2)
# print(labels.iloc[1])
train_image=[]


label_filenames = [(os.path.join(dataset_dir, x))
    for x in os.listdir(dataset_dir)]

number_filenames = [(os.path.join(number_dir, x))
    for x in os.listdir(number_dir)]
#
#img = Image.open(label_filenames[1])
img1 = imread(label_filenames[1])
print(img1.shape)

train_image_set = []


def mse(imageA, imageB):
    # 计算两张图片的MSE指标
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # 返回结果，该值越小越好
    return err
def crop(input_img_path, output_img_path):
    image = Image.open(input_img_path)
    #image_crop = image.resize((876,657))
    image.save(output_img_path)

def imgsimilar(input_img_path, out_img_path ):
    img1 = imread(input_img_path)
    img2 = imread(out_img_path)
    #img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
    m=mse(img1, img2)
    #ssim = compare_ssim(img1, img2, multichannel=True)

    if m == 0:
        #train_image_set.append(out_img_path)
        # train_image_set.append(out_img_path.split('\\')[-1][:-4])
        crop(out_img_path,savedataset_dir+"/"+out_img_path.split('\\')[-1])#保存图像的代码
        #保存label
        for i in range(len(labels)):
            if labels.iloc[i,0][:-4] == out_img_path.split('\\')[-1][:-4]:
                frame = pd.DataFrame([[labels.iloc[i, 0], labels.iloc[i, 1], labels.iloc[i, 2], labels.iloc[i, 3],labels.iloc[i, 4], labels.iloc[i, 5], labels.iloc[i, 6]]],
                                     columns=['file', 'mode', 'x1', 'y1', 'x2', 'y2', 'block'])
                frame.to_csv(csv_file_3, mode='a', header=False)

    return m
num = 0
for path_i in number_filenames:
    for path_j in label_filenames:
        if imgsimilar(path_i, path_j)==0:
            num=num+1
            print(num)
            break

print(train_image_set)




