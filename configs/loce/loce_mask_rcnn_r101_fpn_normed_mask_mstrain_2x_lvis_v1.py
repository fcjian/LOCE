_base_ = './loce_mask_rcnn_r50_fpn_normed_mask_mstrain_2x_lvis_v1.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

load_from = './work_dirs/mask_rcnn_r101_fpn_normed_mask_mstrain_2x_lvis_v1/epoch_24.pth'
