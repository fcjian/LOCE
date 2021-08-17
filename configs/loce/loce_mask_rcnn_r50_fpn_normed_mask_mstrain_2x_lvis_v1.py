_base_ = './mask_rcnn_r50_fpn_normed_mask_mstrain_2x_lvis_v1.py'

model = dict(
    roi_head=dict(
        type='LoceRoIHead',
        bbox_head=dict(
            type='LoceShared2FCBBoxHead',  # Shared2FCBBoxClassfocalHead
            loss_cls=dict(
                type='EquilibriumLoss', use_sigmoid=False, loss_weight=1.0),
            )
    ),
    train_cfg=dict(
        rcnn=dict(
            mfs=dict(
                queue_size=80,
                gpu_statictics=False,
                sampled_num_classes=8,
                sampled_num_features=4
            )
        )
    )
)

evaluation = dict(interval=6, metric=['bbox', 'segm'])

# learning policy
lr_config = dict(
    policy='step',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=0.001,
    step=[3, 5])
total_epochs = 6

# custon hooks
custom_hooks = [
    dict(type="InitializerHook")
]

load_from = './work_dirs/mask_rcnn_r50_fpn_normed_mask_mstrain_2x_lvis_v1/epoch_24.pth'
# load_from = './work_dirs/mask_rcnn_r50_fpn_mstrain_2x_lvis_v1_masknorm/epoch_24.pth'

# Train which part, 0 for all, 1 for fc_cls, fc_reg, rpn and mask_head
selectp = 1
