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

if __name__ == '__main__':
    with open(os.path.join('/home/yiren/artemis-speaker-tools-b/features_path','wikiart_split.pkl'),'rb') as file:
        paints_ids_dict = dict(pickle.load(file))

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    WIKIART_ROOT = '/home/yiren/artemis/wikiart'
    with open('tmp.csv', 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
        for paint_name, id in paints_ids_dict.items():
            ## prepare file_name and id
            im_file = str(WIKIART_ROOT + paint_name + '.jpg')
            image_id = id
            ## writer to results
            writer.writerow(get_detections_from_im(im_file, image_id))
            print(im_file, image_id)
