import torch
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import cv2
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    print("label file: ", args.label_file)
    print("model width: ", args.mb2_width_mult)
    print("args.dataset: ", args.dataset)
    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")
    print("dataset_type: ", args.dataset_type)

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    if args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    print("nms_method: ", args.nms_method)

    if args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    print("CLEAR")

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("image: ", image)
        cv2.imwrite("original11.jpg", image)
        print("image type: ", type(image))
        print("image dtype: ", image.dtype)
        print("image.shape: ", image.shape)

        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        print("image.shape: ", image.shape)

        boxes, labels, probs = predictor.predict(image)
        
        #sys.exit()
        #print("image.shape: ", image.shape)
        ##print("boxes.shape: ", boxes.shape)
        #print("probs.shape: ", probs.shape)
        #print("probs: ", probs)
        new_boxes = boxes[probs >= 0.5]
        #print("new_boxes: ", new_boxes.shape)
        cv2.imwrite("original.jpg", image)
        #break
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        orig_image = image

        boxes = new_boxes
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            #box[0] = box[0].numpy(); box[1] = box[1].numpy(); box[2] = box[2].numpy(); box[3] = box[3].numpy()
            #print(box[0].numpy())
            #print(box[1].numpy())
            #print(box[2].numpy())
            #print(box[3].numpy())
            cv2.rectangle(orig_image, (int(box[0].numpy()), int(box[1].numpy())),
                                      (int(box[2].numpy()), int(box[3].numpy())),
                                      (255, 255, 0), 4)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        (int(box[0].numpy()) + 20, int(box[1].numpy()) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        path = "run_ssd_example_output.jpg"
        cv2.imwrite(path, orig_image)
        print(f"Found {len(probs)} objects. The output image is {path}")
        break
    print("[DEBUG] finish")
