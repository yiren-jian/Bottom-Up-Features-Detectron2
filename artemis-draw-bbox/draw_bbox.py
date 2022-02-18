import sys
import base64
import csv
import pickle

# import some common libraries
import numpy as np
import os, cv2
csv.field_size_limit(sys.maxsize)
from detectron2.config import get_cfg
from detectron2 import model_zoo
import argparse

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images',  help='Total number of images to show',
                        default=10, type=int)
    parser.add_argument('--num_bboxes',  help='Total number of bboxes in an image',
                        default=36, type=int)

    args = parser.parse_args()
    detections_path = '/home/yiren/artemis-speaker-tools-b/features_path2/'
    with open(os.path.join('/home/yiren/artemis-speaker-tools-b/features_path','wikiart_split.pkl'),'rb') as file:
        paints_ids_dict = dict(pickle.load(file))

    imageid_address = {}
    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
    WIKIART_ROOT = '/home/yiren/artemis/wikiart'
    for paint_name, id in paints_ids_dict.items():
        ## prepare file_name and id
        im_file = str(WIKIART_ROOT + paint_name + '.jpg')
        image_id = id
        ## writer to results
        imageid_address[str(id)] = im_file


    n = 0
    image_info = dict()
    with open(os.path.join(detections_path, 'tmp.csv'), "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                data = item[field]
                # buf = base64.decodestring(data)
                buf = base64.b64decode(data[1:])
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'], -1))
            image_info[item['image_id']] = {}
            image_info[item['image_id']]['num_boxes'] = item['num_boxes']
            image_info[item['image_id']]['image_h'] = item['image_h']
            image_info[item['image_id']]['image_w'] = item['image_w']
            image_info[item['image_id']]['bbx'] = item['boxes'].tolist()
            image_info[item['image_id']]['address'] = imageid_address[str(item['image_id'])]
            n += 1
            if n == args.num_images:
                break

    color = (0, 0, 255) #red

    n_bbox = 0
    for i in image_info:
        nnb = 0
        im = cv2.imread(image_info[i]['address'])
        import detectron2.data.transforms as T

        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        im = aug.get_transform(im).apply_image(im)
        for i_n in range(image_info[i]['num_boxes']):
            start_point = (round(image_info[i]['bbx'][i_n][0]), round(image_info[i]['bbx'][i_n][1]))
            end_point = (round(image_info[i]['bbx'][i_n][2]), round(image_info[i]['bbx'][i_n][3]))
            cv2.rectangle(im, start_point, end_point, color, 2)

            n_bbox += 1
            if n_bbox == args.num_bboxes:
                break

        cv2.imwrite(str(i) + '.jpg', im)
