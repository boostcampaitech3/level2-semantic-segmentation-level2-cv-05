"""
get semantic segmentation annotations from coco data set.
"""
from re import L
from urllib.parse import ParseResultBytes
from PIL import Image
import imgviz # pip install imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import shutil
import numpy as np
 
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8)) 
    lbl_pil.save(save_path)

 
def main(args):
    annotation_file = os.path.join(args.input_dir, '{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'copy_paste/images'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'copy_paste/annotations'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for index, imgId in tqdm.tqdm(enumerate(imgIds), total=len(imgIds)):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask[coco.annToMask(anns[i+1]) == 1] = anns[i+1]['category_id']
                # mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            img_origin_path = os.path.join(args.input_dir, img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'copy_paste/images', f'{index:04}.jpg')
            seg_output_path = os.path.join(args.input_dir, 'copy_paste/annotations', f'{index:04}.png')
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/opt/ml/input/data", type=str,
                        help="coco dataset directory")
    parser.add_argument("--split", default="train", type=str,
                        help="train2017 or val2017")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)