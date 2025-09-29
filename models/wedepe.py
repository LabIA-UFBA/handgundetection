from PIL import Image
import torchvision.ops as ops
import torch
import matplotlib.pyplot as plt
from .adapted_wedepe import AdaptedWeDePe


class WeDePe(AdaptedWeDePe):
    def __init__(self, detection_model, pose_model):
        super().__init__(detection_model, pose_model)

    def predict(self, img, conf=0.25, iou=0.7):
        poses = self.pose_model(img, verbose=False)[0].keypoints.xy
        poses = self.add_keypoints(poses)
        boxes = self.get_boxes(poses)
        crops = self.crop_image(img, boxes)
        detections = self.detect(img, crops, conf, iou)

        return detections
    
    def detect(self, img: Image, crops, conf, iou):
        width, height = img.size
        
        C = 2
        R = 1
        while C * R < len(crops):
            if C < (R * 2):
                C += 1
            else:
                R += 1

        grid_img = Image.new("RGB", (width, height), (0, 0, 0))
        grid_size = min(height // R, width // C)

        for crop in crops:
            crop.append(crop[1].size[0])

        for i in range(len(crops)):
            w = (i % C) * grid_size
            h = (i // C) * grid_size

            crop_img = crops[i][1].copy()
            crop_img = crop_img.resize((grid_size, grid_size))
            grid_img.paste(crop_img, (w, h))

        results = self.detection_model(grid_img, conf=conf, iou=iou, verbose=False)[0].boxes.data.clone()
        detections = []
        
        for result in results:
            try:
                x = (result[0] + result[2]) / 2 // grid_size
                y = (result[1] + result[3]) / 2 // grid_size
                crop_idx = int(y * C + x)
                scale = crops[crop_idx][2] / grid_size

                result[0] = (result[0] - x * grid_size).clamp_(0, grid_size) * scale + crops[crop_idx][0][0]
                result[1] = (result[1] - y * grid_size).clamp_(0, grid_size) * scale + crops[crop_idx][0][1]
                result[2] = (result[2] - x * grid_size).clamp_(0, grid_size) * scale + crops[crop_idx][0][0]
                result[3] = (result[3] - y * grid_size).clamp_(0, grid_size) * scale + crops[crop_idx][0][1]

                detections.append(result)
            except:
                pass

        if not detections:
            return torch.empty((0, 6))
    
        detections = torch.stack(detections, dim=0)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        keep = ops.nms(boxes, scores, 0.3)
        detections = detections[keep]

        return detections