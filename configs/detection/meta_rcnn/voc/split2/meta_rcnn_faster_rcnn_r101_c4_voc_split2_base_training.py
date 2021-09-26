_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../meta_rcnn_faster_rcnn_r50_c4.py',
    '../../../_base_/default_runtime.py'
]
# Predefined ann_cfg, classes and class_splits are defined in
# mmfewshot.detection.datasets.few_shot_data_config
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT2'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT2')),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'),
    model_init=dict(classes='BASE_CLASSES_SPLIT2'))
lr_config = dict(warmup=None, step=[12000, 18000])
evaluation = dict(interval=30000)
checkpoint_config = dict(interval=6000)
runner = dict(max_iters=24000)
optimizer = dict(lr=0.004)
# model settings
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(num_classes=15, num_meta_classes=15),
        shared_head=dict(pretrained=pretrained),
    ))