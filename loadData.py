# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 09:43:14 2018

@author: MJ
"""

import numpy as np
import struct
import matplotlib.pyplot as plt


def readfile():
    with open('.\\mnist\\train-images.idx3-ubyte','rb') as f1:
        buf1 = f1.read()
    with open('.\\mnist\\train-labels.idx1-ubyte','rb') as f2:
        buf2 = f2.read()
    return buf1, buf2
def read_test_file():
    
    with open('.\\mnist\\t10k-images.idx3-ubyte','rb') as t1:
        but1=t1.read()
    with open('.\\mnist\\t10k-labels.idx1-ubyte','rb') as t2:
        but2=t2.read()
    return but1,but2

        


#def get_image(buf1, nums):
#    image_index = 0
#    image_index += struct.calcsize('>IIII')
#    img = []
#    for i in range(nums):
#        temp = struct.unpack_from('>784B', buf1, image_index) # '>784B'的意思就是用大端法读取784个unsigned byte
#        img.append(np.reshape(temp,(28,28)))
#        image_index += struct.calcsize('>784B')  # 每次增加784B
#    return img

def get_image(buf1, nums):
    image_index = 0
    image_index += struct.calcsize('>IIII')
    img = np.zeros([nums, 784], int)
    for i in range(nums):
        temp = struct.unpack_from('>784B', buf1, image_index) # '>784B'的意思就是用大端法读取784个unsigned byte
        img[i, :] = temp
        image_index += struct.calcsize('>784B')  # 每次增加784B
    return img


def get_label(buf2, nums): # 得到标签数据
    label_index = 0
    label_index += struct.calcsize('>II')

    return struct.unpack_from('>'+str(nums)+'B', buf2, label_index)


if __name__ == "__main__":
    image_data, label_data = readfile()
    im = get_image(image_data, 15)
    label = get_label(label_data, 15)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        title = u"标签对应为："+ str(label[i+5])
        plt.title(title, fontproperties='SimHei')
        plt.imshow(np.reshape(im[i+5,:],(28,28)), cmap='gray')
    plt.show()
