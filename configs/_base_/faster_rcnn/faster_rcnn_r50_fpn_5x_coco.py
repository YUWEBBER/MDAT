_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_ciou_3x.py',
    '../_base_/datasets/SSDD.py',
    '../_base_/schedules/schedule_5x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='CIoULoss', loss_weight=12.0))))
