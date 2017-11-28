# Semantic Segmentation Using [Pytorch](http://pytorch.org/)

## Network

- [x] [FCN-8s](https://arxiv.org/abs/1605.06211)
- [x] [FCN-16s](https://arxiv.org/abs/1605.06211)
- [x] [FCN-32s](https://arxiv.org/abs/1605.06211)
- [x] [DeepLab](https://arxiv.org/abs/1412.7062) (deeplab v1)
- [x] [DeepLab-LargeFOV](https://arxiv.org/abs/1412.7062) (deeplab v1)
- [x] [Deeplab-MSc-LargeFOV](https://arxiv.org/abs/1412.7062) (deeplab v1)
- [x] [U-Net](https://arxiv.org/abs/1505.04597) (based on VGG16)
## Metric on PASCAL VOC 2012 validation set
Network|mean IoU|Download
:---:|:---:|:---:
DeepLab|63.04 (64.38 in paper)|[deeplab.pth]()

## Dataset

### [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#citation)
The original dataset contains 1464 images for training and 1449 images for validation. The dataset is augmented by the extra annotations provided by [Hariharan et al](http://home.bharathh.info/pubs/codes/SBD/download.html), resulting in 10582 images for training and 1449 images for validation(note that the validation images remain the same).
