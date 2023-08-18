# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.

Example:
    python demo/demo_attention_rpn_detector_inference.py \
        ./demo/demo_detection_images/query_images/demo_query.jpg
        configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py
        ./work_dirs/attention-rpn_r50_c4_4xb2_coco-base-training/latest.pth
"""  # nowq

import os
import random
import xmltodict
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)

# TODO: read VOC classes and base directory from config file.
VOC_CLASSES_LIST = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor']
VOC_BASE_DIR = "/home/DIINF/cloyolav/tesis/code/mmfewshot-tesis/data/VOCdevkit/VOC2007"
IMAGES_SETS_DIR = os.path.join(VOC_BASE_DIR, "ImageSets", "Main")
ANNOTATIONS_DIR = os.path.join(VOC_BASE_DIR, "Annotations")
IMAGES_DIR = os.path.join(VOC_BASE_DIR, "JPEGImages")
SUPPORT_BASE_DIR = "/home/DIINF/cloyolav/tesis/code/voc_supports/supports"


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    # parser.add_argument('image', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # parser.add_argument(
    #     '--support-images-dir',
    #     default='demo/demo_detection_images/support_images',
    #     help='Image file')
    args = parser.parse_args()
    return args


def get_ann_from_id(img_id):
    '''
    Return VOC annotation (as a dictionary) from image ID.
    '''
    img_fn = os.path.join(IMAGES_DIR, img_id + ".jpg")
    # load annotation
    img_ann_fn = os.path.join(ANNOTATIONS_DIR, img_id + ".xml")
    with open(img_ann_fn) as file:
        # read file contents
        file_data = file.read()
        # parse data using package
        ann_data = xmltodict.parse(file_data)
    return ann_data['annotation']


def generate_voc_support_dict(support_base_dir, classes, use_difficult=False):
    '''
    Create a dictionary with VOC classes as keys, and a list with their
    respective support filenamenes as values.
    '''
    print("Generating VOC support dict...")
    support_fn_dict = {}
    for class_name in classes:
        class_dir = os.path.join(support_base_dir, class_name)
        support_fn_list = sorted(os.listdir(class_dir))
        if not use_difficult:
            support_fn_list = [fn for fn in support_fn_list if 'diff' not in fn]
        support_fn_dict[class_name] = support_fn_list
    total = 0
    for k, v in support_fn_dict.items():
        print(f"{k}: {len(v)} instances")
        total += len(v)
    print(f"Total: {total}")
    return support_fn_dict


def generate_query_support_pairs(imgs_ids, support_base_dir, classes_list,
                                 n_supports=5, use_difficult=False):
    '''
    For each image, obtain its gt_labels and gt_bboxes, and also sample
    n_support examples.
    '''
    support_fn_dict = generate_voc_support_dict(support_base_dir, classes_list,
                                                use_difficult)
    imgs_dict = {}
    for img_id in imgs_ids:
        ann = get_ann_from_id(img_id)
        objs_info = ann['object']
        if type(objs_info) is not list:
            objs_info = [objs_info]
        gt_labels = []
        gt_bboxes = []
        for obj_info in objs_info:
            # Filter out difficult instances
            if obj_info['difficult'] == '0':
                gt_labels.append(obj_info['name'])
                coords = obj_info['bndbox']
                # TODO: use [xmin, ymin, w, h] format instead?
                gt_bboxes.append([int(coords['xmin']), int(coords['ymin']),
                                  int(coords['xmax']), int(coords['ymax'])])
        # Consider only images with at least 1 not difficult instance
        if len(gt_labels) > 0:
            # For each class, sample n supports
            random.seed(int(img_id))  # set random seed to reproduce the results
            sampled_supports_dict = {}
            gt_labels_unique = set(gt_labels)
            for class_name in gt_labels_unique:
                # print(f"  class_name = {class_name}")
                support_fn_list = support_fn_dict[class_name]
                sampled_supports = []
                # Don't use supports obtained from this query
                while len(sampled_supports) < n_supports:
                    sampled_support = random.choice(support_fn_list)
                    sampled_support_id = sampled_support.split("_")[0]
                    if sampled_support_id != img_id:
                        sampled_supports.append(sampled_support)
                sampled_supports_dict[class_name] = sampled_supports
            imgs_dict[img_id] = {'gt_labels': gt_labels,
                                 'gt_bboxes': gt_bboxes,
                                 'sampled_supports': sampled_supports_dict}
    return imgs_dict


def main(args):
    # build the model from a config file and a checkpoint file
    print("Calling init_detector...")
    # mmfewshot/detection/apis/inference.py
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # prepare support images, each demo image only contain one instance
    print("Model initialized!")
    '''
    files = os.listdir(args.support_images_dir)
    support_images = [
        os.path.join(args.support_images_dir, file) for file in files
    ]
    classes = [file.split('.')[0] for file in files]
    support_labels = [[file.split('.')[0]] for file in files]
    '''
    # get all test images with its gt info and sample supports for each class
    print("Generating query / support pairs dictionary...")
    test_fn = os.path.join(IMAGES_SETS_DIR, "test.txt")
    with open(test_fn) as f:
        imgs_ids = sorted(f.readlines())
    imgs_ids = [img_id.strip() for img_id in imgs_ids]
    imgs_dict = generate_query_support_pairs(imgs_ids, SUPPORT_BASE_DIR,
                                             VOC_CLASSES_LIST, n_supports=5,
                                             use_difficult=False)
    # make inference with all the images and supports from imgs_dict
    for img_id, img_info in [list(imgs_dict.items())[0]]:
        img_fn = os.path.join(IMAGES_DIR, img_id + ".jpg")
        print(f"img_id = {img_id}")
        for class_name, supports in img_info['sampled_supports'].items():
            print(f"  class_name = {class_name}")
            for support_fn in supports:
                support_images = [os.path.join(SUPPORT_BASE_DIR, class_name, support_fn)]
                support_labels = [[class_name]]
                classes = [class_name]
                print(f"    Processing {support_fn}...")
                process_support_images(model, support_images, support_labels,
                                       classes=classes)
                # mmfewshot/detection/apis/inference.py
                # It calls to foward_test in BaseDetector, which calls to simple_test in AttentionRPNDetector
                print("    Calling inference_detector...")
                result = inference_detector(model, img_fn)

                print(f"    Before thr filter: {result[0].shape}")
                print(f"    {result[0][:10]}")
                # Filter by confidence threshold
                # result = result[0]
                # result = [result[result[:, 4] > 0.9]]
                # print(f"After thr filter: {result[0].shape}")
                print("    Inference done!")
                # show the results
                output_fn = f"{img_id}_{support_fn}"
                output_fn = os.path.join(SUPPORT_BASE_DIR, output_fn)
                show_result_pyplot(model, img_fn, result, score_thr=args.score_thr, out_file=output_fn)

if __name__ == '__main__':
    args = parse_args()
    main(args)
