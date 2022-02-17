# bottom-up-features
This repo covers the implementation of extracting features of images for training [Mesh-Memory-Transformer](https://github.com/aimagelab/meshed-memory-transformer). The features are extracted by pre-trained ResNet101-faster-RCNN. It selects top 36 detections and each detection has a feature of dimension 2048. Thus, for each image, the feature is a tensor of dimension 36x2048.

Here is an example of preparing features for training on [ArtEmis](https://github.com/Kilichbek/artemis-speaker-tools-b)

## Requirements
We tested on a Nvidia RTX-A6000 with pytorch-1.10, cuda-11.3.
```
# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# install Detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## Extract features
If you have a single GPU on the system, simply do:
```
python generate_tsv.py
```
If you have multiple GPUs (for example, 4 GPUs):
```
python generate_tsv_by_gpu.py --num_gpus=4 --gpu=[0|1|2|3] --split_file='wikiart_split.pkl' --wikiart_root='/home/yiren/artemis/wikiart'
```
where `wikiart_split.pkl` stores the image file names and image ids for the dataset. Running above commands will generate tsv files `tmp0.csv`, `tmp1.csv`, `tmp2.csv`, and `tmp3.csv`. Then, call:
```
python merge_tsv.py --num_gpus=4
```

## Acknowledgment
Thanks the author for the original [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) implementation and models from [Detectron2](https://github.com/facebookresearch/detectron2).
