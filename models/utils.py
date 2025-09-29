import torch

def box_iod(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    # Compute intersection coordinates
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # max(x1)
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # max(y1)
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # min(x2)
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # min(y2)

    # Compute intersection area
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    # Compute area of detection boxes (boxes2)
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area_boxes2 = area_boxes2[None, :]  # Reshape for broadcasting

    # Compute IoD
    iod = inter_area / (area_boxes2 + 1e-6)  # Add small value to prevent division by zero

    return iod


def box_iogt(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    # Compute intersection coordinates
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # max(x1)
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # max(y1)
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # min(x2)
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # min(y2)

    # Compute intersection area
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h

    # Compute area of ground truth boxes (boxes1)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes1 = area_boxes1[:, None]  # Reshape for broadcasting

    # Compute IoGT
    iogt = inter_area / (area_boxes1 + 1e-6)  # Add small value to prevent division by zero

    return iogt
