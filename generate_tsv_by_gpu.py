### modified from https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py

import os, sys
import base64
import numpy as np
import cv2
import csv
import pickle
from pathlib import Path

from utils import get_detections_from_im
csv.field_size_limit(sys.maxsize)

import argparse

if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--num_gpus',  help='Total number of GPUs in the system',
                        default=4, type=int)
    parser.add_argument('--gpu',  help='GPU id(s) to use',
                        default=0, type=int)
    parser.add_argument('--split_file', help='wikiart_split.pkl',
                        default='/home/yiren/artemis-speaker-tools-b/features_path/wikiart_split.pkl', type=str)
    parser.add_argument('--wikiart_root', help='Root directory for images (e.g., WikiArt)',
                        default='/home/yiren/artemis/wikiart', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    with open(args.split_file,'rb') as file:
        paints_ids_dict = dict(pickle.load(file))

    all_ids = [i for i in range(len(paints_ids_dict))]
    selected_ids = np.array_split(all_ids, 4)[args.gpu].tolist()

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    WIKIART_ROOT = args.wikiart_root
    with open('tmp%d.csv'%(args.gpu), 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
        for paint_name, id in paints_ids_dict.items():
            if id in selected_ids:
                ## prepare file_name and id
                im_file = str(WIKIART_ROOT + paint_name + '.jpg')
                image_id = id
                ## writer to results
                writer.writerow(get_detections_from_im(im_file, image_id))
                print(im_file, image_id)
