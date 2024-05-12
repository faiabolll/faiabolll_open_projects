from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

import os
from glob import glob
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2 as cv
import numpy as np
import pandas as pd
%matplotlib inline

BINARY_IMAGE_PATH = 'data\\images\\mask\\'
COLOR_MASK_IMAGE_PATH = 'data\\images\\mask colored\\'
ORIGINAL_IMAGE_PATH = 'data\\images\\original'
# flags = [i for i in dir(cv) if i.startswith('COLOR_') and 'hsv' in i.lower()]
# print( flags )

def trim_binary_image(img):
    if img.max() == 0:
        print('Zero-image')
        return (0,0,0,0)
    """Accepts only binarized images else result unpredictable"""
    horizCropArr = np.argmax(img, axis=1)
    lcrop = horizCropArr[horizCropArr != 0].min()

    vertCropArr = np.argmax(img, axis=0)
    ucrop = vertCropArr[vertCropArr != 0].min() if img[0,:].max() == 0 else 0

    # Rotating image by 180 degrees
    img_rot = np.rot90(img, 2)

    horizCropArr_rot = np.argmax(img_rot, axis=1)
    # rcrop = horizCropArr_rot[horizCropArr_rot != 0].min() if img[:,0].max() == 0 else 0
    rcrop = horizCropArr_rot[horizCropArr_rot != 0].min()
    rcrop = img.shape[1] - rcrop

    vertCropArr_rot = np.argmax(img_rot, axis=0)
    if img_rot[0,:].max() == 0:
        dcrop = vertCropArr_rot[vertCropArr_rot != 0].min()
    else:
        dcrop = 0
    dcrop = img.shape[0] - dcrop

    return (lcrop, ucrop, rcrop, dcrop)
    # return img[ucrop:dcrop, lcrop:rcrop]

def apply_mask(orig, mask):
    img = orig.copy()
    for i in range(3):
        img[:,:,i:i+1] = np.minimum(img[:,:,i:i+1], np.expand_dims(mask, axis=2))
    return img


def read_binary_image(image_num):
    bin_mask = Image.open(os.path.join(BINARY_IMAGE_PATH, f'{image_num}.png'), mode='r').convert('L')
    bin_mask = bin_mask.point(lambda p: 255 if p > 128 else 0)
    bin_mask_np = np.array(bin_mask, dtype=np.uint8)
    return bin_mask_np


def read_color_mask_image(image_num):
    img = cv.imread(os.path.join(COLOR_MASK_IMAGE_PATH, f'{image_num}.exr'), cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) * 5
    bin_mask_np = read_binary_image(image_num)
    img = apply_mask(img,bin_mask_np)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    return img

def labeling(image_num):
    img = read_color_mask_image(image_num)
    width, height = img.shape[0], img.shape[1]

    h = img[:,:,0] # H channel of HSV image

    freqs = pd.Series(h[h>0]).apply(lambda x: round(x,2)).value_counts()
    freqs = freqs[freqs > 50]

    h_rounded = np.around(h.copy(), 2)

    to_draw = img.copy()
    labels = []
    for hh in freqs.index:
        h_rounded_ge = h_rounded >= hh - 0.1
        h_rounded_le = h_rounded <= hh + 0.1
        res_masked = np.logical_and(h_rounded_ge, h_rounded_le).astype(np.uint8)

        rectangle = trim_binary_image(res_masked)

        l,u,r,d = rectangle
        x, y = l + (r-l) / 2, u + (d-u) / 2
        w, h = (r-l), (d-u)
        S = w * h

        # skip tiny objects
        if S < 1200:
            continue

        # saving labels
        x_rel, y_rel, w_rel, h_rel = x / width, y / height, w / width, h / height
        label = f'{x_rel} {y_rel} {w_rel} {h_rel}'
        labels.append(label)

        # put rectangle and label to this rectangle
        # cv.rectangle(to_draw, (l,u), (r,d), color=(255,0,0), thickness=2)
        # cv.putText(to_draw, f"{S:.2f}", (l,u), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(1,1,0))

    with open(f'data\\labels\\{str(image_num).zfill(6)}.txt', 'w') as f:
        f.write('\n'.join(labels))

    # plt.imshow(to_draw)      
    
def create_labels(images_path='data\\images\\original'):
    img_names = glob(images_path + '\\*')
    img_names = [name.split('\\')[-1].split('.')[0] for name in img_names]
    for img_name in img_names:
        labeling(img_name)
    
def main():
    create_labels()
    
if __name__ == '__main__':
    main()