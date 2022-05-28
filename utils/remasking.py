import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


dst_root = ''
data_root = ''


def inner_check(inner_patch, p=0.4):
    size = inner_patch.shape
    counter = np.bincount(inner_patch)
    return counter[0] / size < p


def most_pixel(out_patch):
    counter = np.bincount(out_patch)
    return np.argmax(counter)


def remasking(origin_mask, kernel_size=5):
    temp_mask = origin_mask.copy()
    ret = origin_mask.copy()
    temp_mask = np.pad(temp_mask, ((2, 2), (2, 2)), 'constant', constant_values=0)
    kernel_length = kernel_size // 2
    for i in range(2, 514):
        for j in range(2, 514):
            if temp_mask[j, i] == 0:
                inner_flag = inner_check(temp_mask[j - 1:j + 2, i - 1:i + 2].flatten())
                if inner_flag:
                    ret[j - kernel_length, i - kernel_length] = most_pixel(
                        temp_mask[j - kernel_length:j + kernel_length + 1,
                        i - kernel_length:i + kernel_length + 1].flatten())
    return ret


def processing(path):
    mask = np.array(Image.open(path))
    remasked = remasking(mask)
    fname = path.split('/')[-1]
    cv2.imwrite(os.path.join(dst_root, fname), remasked)
    return True


def main():
    global data_root, dst_root
    data_root = '/opt/ml/input/data/mmseg_fix/annotations/training'
    mask_paths = glob(f'{data_root}/*.png')
    dst_root = '/opt/ml/input/data/mmseg_remasking/annotations/training'

    os.makedirs(dst_root, exist_ok=True)
    with tqdm(total=len(mask_paths)) as pbar:
        with ProcessPoolExecutor(max_workers=7) as executor:
            dict_future = [executor.submit(processing, path) for path in mask_paths]
            for future in as_completed(dict_future):
                pbar.update(1)


if __name__ == "__main__":
    main()
