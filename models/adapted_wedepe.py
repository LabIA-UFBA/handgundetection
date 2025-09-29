from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import torchvision.ops as ops
import matplotlib.pyplot as plt

from .base_model import BaseModel

class AdaptedWeDePe(BaseModel):
    def __init__(self, detection_model, pose_model: YOLO):
        super().__init__(detection_model)

        self.pose_model = YOLO(pose_model)
        self.keys = {
            'Nose': 0, 'Left Eye': 1, 'Right Eye': 2, 'Left Ear': 3, 'Right Ear': 4, 'Left Shoulder': 5, 'Right Shoulder': 6,
            'Left Elbow': 7, 'Right Elbow': 8, 'Left Wrist': 9,  'Right Wrist': 10,  'Left Hip': 11, 'Right Hip': 12, 'Left Knee': 13,
            'Right Knee': 14, 'Left Ankle': 15, 'Right Ankle:': 16, 'Chest': 17, 'Left Hand': 18, 'Right Hand': 19
        }
        self.limbs = [(0, 17), (5, 17), (6, 17), (7, 5), (8, 6), (9, 7), (10, 8), (11, 17), (12, 17)]

    def predict(self, img, conf=0.25, iou=0.7):
        poses = self.pose_model(img, verbose=False)[0].keypoints.xy
        poses = self.add_keypoints(poses)
        boxes = self.get_boxes(poses)
        crops = self.crop_image(img, boxes)
        detections = self.detect(crops, conf, iou)

        return detections
    
    def add_keypoints(self, poses: torch.Tensor):
        if poses.numel() == 0:  # Check if poses is empty
            return torch.empty((0, 20, 2), dtype=poses.dtype, device=poses.device)  # Ensure correct shape
        
        shoulders = poses[:, [self.keys['Left Shoulder'], self.keys['Right Shoulder']]]
        chest = shoulders.mean(dim=1, keepdim=True) * (shoulders != 0).all(dim=1, keepdim=True)

        wrists = poses[:, [self.keys['Left Wrist'], self.keys['Right Wrist']]]
        elbows = poses[:, [self.keys['Left Elbow'], self.keys['Right Elbow']]]
        hands = wrists + 0.3 * (wrists - elbows) * (wrists != 0).all(dim=1, keepdim=True)

        return torch.cat((poses, chest, hands), dim=1)
    
    def get_boxes(self, poses):
        boxes = []
        for pose in poses:
            limbs = pose[self.limbs, :]
            limbs = limbs[(limbs != 0).all(dim=(1,2))]
            limb_lengths = torch.norm(limbs[:, 0] - limbs[:, 1], dim=1)

            apf = (limb_lengths * 0.5 / len(limb_lengths)).sum() * 1.5

            left_hand = pose[self.keys['Left Hand']]
            if (left_hand != 0).any().item():
                box = torch.tensor([left_hand[0] - apf, left_hand[1] - apf, left_hand[0] + apf, left_hand[1] + apf])
                boxes.append(box)

            right_hand = pose[self.keys['Right Hand']]
            if (right_hand != 0).any().item():
                box = torch.tensor([right_hand[0] - apf, right_hand[1] - apf, right_hand[0] + apf, right_hand[1] + apf])
                boxes.append(box)

        if not boxes:
            return torch.empty((0, 4))

        return torch.stack(boxes, dim=0)
    
    def crop_image(self, img: Image, boxes: torch.Tensor):
        crops = []

        for box in boxes:
            box = tuple(box.round().int().tolist())
            crops.append([box[:2], img.crop(box)])

        return crops
    
    def detect(self, crops, conf=0.25, iou=0.7):
        detections = []

        for (x, y), img in crops:
            result = self.detection_model(img, conf=conf, iou=iou, imgsz=320, verbose=False)[0].boxes.data
            result = result + torch.tensor([x, y, x, y, 0, 0]).to(result.device)
            detections.append(result)

        if not detections:
            return torch.empty((0, 6))
    
        detections = torch.cat(detections)

        boxes = detections[:, :4]
        scores = detections[:, 4]
        keep = ops.nms(boxes, scores, 0.3)
        detections = detections[keep]  

        return detections

    def plot(self, img, detections):
        draw = ImageDraw.Draw(img)

        for detection in detections:
            detection = detection[:4].round().int().tolist()
            draw.rectangle(detection, outline="red", width=10)

        return img