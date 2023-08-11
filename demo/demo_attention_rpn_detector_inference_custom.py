# Copyright (c) OpenMMLab. All rights reserved.
"""Inference Attention RPN Detector with support instances.

Example:
    python demo/demo_attention_rpn_detector_inference.py \
        ./demo/demo_detection_images/query_images/demo_query.jpg
        configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_base-training.py
        ./work_dirs/attention-rpn_r50_c4_4xb2_coco-base-training/latest.pth
"""  # nowq

import os
from argparse import ArgumentParser

from mmdet.apis import show_result_pyplot

from mmfewshot.detection.apis import (inference_detector, init_detector,
                                      process_support_images)


def parse_args():
    parser = ArgumentParser('attention rpn inference.')
    # parser.add_argument('image', help='Image file')
    parser.add_argument('base_dir', help='Base directory')
    # parser.add_argument('-class_type', help='seen / unseen during training')
    # parser.add_argument('-class_name', help='Support class name')
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


def main(args):
    classes_dict = {'seen': ['bicycle', 'bird', 'boat','bottle', 'bus', 'car',
                             'chair', 'diningtable', 'dog', 'horse', 'motorbike',
                             'person', 'pottedplant', 'sofa', 'train', 'tvmonitor'],
                    'unseen': ['aeroplane', 'cat', 'cow', 'sheep']}
    # build the model from a config file and a checkpoint file
    print("Calling init_detector...")
    # mmfewshot/detection/apis/inference.py
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("Model initialized!")

    # for class_type in classes_dict:
    for class_type in ['unseen']:
        print(f"class_type = {class_type}:")
        for class_name in classes_dict[class_type]:
            print(f"  class_name = {class_name}")
            # prepare support images, each demo image only contain one instance
            class_dir = os.path.join(args.base_dir, class_type, class_name)
            image_dir = os.path.join(class_dir, "query")
            image_fn = os.listdir(image_dir)[0]
            image_fn = os.path.join(image_dir, image_fn)
            print(image_fn)
            support_images_dir = os.path.join(class_dir, "support")
            files = os.listdir(support_images_dir)
            support_images = [
                os.path.join(support_images_dir, file) for file in files
            ]
            print(support_images)
            classes = [file.split('.')[0] for file in files]
            support_labels = [[file.split('.')[0]] for file in files]
            print("Processing support images...")
            # mmfewshot/detection/apis/inference.py
            process_support_images(
                model, support_images, support_labels, classes=classes)
            print("Support images processed!")
            # test a single image
            # mmfewshot/detection/apis/inference.py
            # It calls to foward_test in BaseDetector, which calls to simple_test in AttentionRPNDetector
            print("Calling inference_detector...")
            result = inference_detector(model, image_fn)
            print(f"Before thr filter: {result[0].shape}")
            print(f"  {result[0][:10]}")
            # Filter by confidence threshold
            # result = result[0]
            # result = [result[result[:, 4] > 0.9]]
            print(f"After thr filter: {result[0].shape}")
            print("Inference done!")
            # show the results
            output_fn = os.path.join(class_dir, "result_full.png")
            show_result_pyplot(model, image_fn, result, score_thr=args.score_thr, out_file=output_fn)

if __name__ == '__main__':
    args = parse_args()
    main(args)
