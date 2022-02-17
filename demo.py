import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


from detectron2.structures import Boxes, Instances
from detectron2.layers import batched_nms
### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L117
def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return boxes, filter_inds[:, 0]


if __name__ == '__main__':

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py#L276
    original_image= cv2.imread('demo.jpg')
    height, width = original_image.shape[:2]

    with torch.no_grad():
        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py#L307
        import detectron2.data.transforms as T
        aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}

        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py#L199
        batched_inputs = [inputs]
        images = model.preprocess_image(batched_inputs)
        features = model.backbone(images.tensor)

        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py#L204
        proposals, _ = model.proposal_generator(images, features, None)
        # results, _ = model.roi_heads(images, features, proposals, None)

        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/roi_heads.py#L476
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = model.roi_heads._shared_roi_transform(
            [features[f] for f in model.roi_heads.in_features], proposal_boxes
        )
        box_features = box_features.mean(dim=[2, 3])
        predictions = model.roi_heads.box_predictor(box_features)

        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/roi_heads.py#L498
        ### pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)
        boxes = model.roi_heads.box_predictor.predict_boxes(predictions, proposals)
        scores = model.roi_heads.box_predictor.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        score_thresh = model.roi_heads.box_predictor.test_score_thresh
        nms_thresh = model.roi_heads.box_predictor.test_nms_thresh
        topk_per_image = 36

        ### https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L117
        result_per_image = [
            fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
        ]
        cls_boxes = result_per_image[0][0]
        box_features = box_features[result_per_image[0][-1]]

        #### check results
        print(box_features.shape)
