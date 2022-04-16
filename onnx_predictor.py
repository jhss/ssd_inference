import numpy as np
import onnxruntime as ort
import base64
import io
import cv2
import sys
import torch

from PIL import Image
from scipy.special import softmax

from vision.ssd.data_preprocessing import PredictionTransform
from utils.timing import timing
from utils import box_utils

class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.transform = PredictionTransform(300,
                                            [127] * 3,
                                            128.0)
        self.filter_threshold = 0.01
        self.iou_threshold = 0.45
        self.nms_method = 'hard'
        self.sigma = 0.5
        self.candidate_size = 200
        self.top_k = -1
        self.class_names = [name.strip() for name in open("./models/voc-model-labels.txt").readlines()]
    @timing
    def predict(self, image_bytes):
        #print(image_bytes)

        img_b64dec = base64.b64decode(image_bytes)
        img_byteIO = io.BytesIO(img_b64dec)
        image = np.array(Image.open(img_byteIO))

        image = cv2.imread('./original11.jpg')

        #cv2.imwrite('converted.jpg', image)
        print(image)
        #sys.exit()
        tensor_image = self.transform(image)
        height, width, _ = image.shape
        #print("image.shape: ", tensor_image.shape)
        #print("height, width: ", height, width)
        #sys.exit()
        ort_inputs = {
            "image": tensor_image.unsqueeze(0).numpy()
        }
        scores, boxes = self.ort_session.run(None, ort_inputs)
        boxes = torch.Tensor(boxes[0])
        scores = torch.Tensor(scores[0])
        print("boxes: ", boxes)
        print("score: ", scores)
        prob_threshold = self.filter_threshold
        '''
        same until here
        '''
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=self.top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        boxes, labels, probs = picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
        print("width, height: ", width, height)
        print("boxes: ", boxes)
        boxes = boxes[probs >= 0.5]
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image, (int(box[0].numpy()), int(box[1].numpy())),
                                 (int(box[2].numpy()), int(box[3].numpy())),
                                 (255, 255, 0), 4)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(image, label,
                        (int(box[0].numpy()) + 20, int(box[1].numpy()) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        path = "onnx_output.jpg"
        cv2.imwrite(path, image)

        '''
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": float(score)})
        print(predictions)
        '''
        return "success"


if __name__ == "__main__":
    image_bytes = open('./base64_test.txt', 'rb').read()
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(image_bytes))
