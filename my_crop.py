from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop(input_img_path, output_img_path):
    image = Image.open(input_img_path)
    image_crop = image.resize((876,657))
    image_crop.save(output_img_path)

dataset_dir = 'D:/浏览器下载/valid'
output_dir = 'D:/浏览器下载/valid768x576'

#获得需要转化的图片路径并生成目标路径
image_filenames = [(os.path.join(dataset_dir, x), os.path.join(output_dir, x))
                       for x in os.listdir(dataset_dir)]
print(image_filenames)
for path in image_filenames:
    crop(path[0], path[1])
