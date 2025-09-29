from ultralytics import YOLO
import pandas as pd
import torch
import torchvision.ops as ops
import cv2
from .dataset import YoloDataset
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc

class BaseModel:
    def __init__(self, detection_model):
        self.detection_model = YOLO(detection_model)

    def predict(self, img, conf=0.25, iou=0.7):
        results = self.detection_model(img, conf=conf, iou=iou, classes=[0], verbose=False)[0].boxes.data

        return results
    
    def val(self, folder, conf=0.25, iou=0.7, ticks=100, func=ops.box_iou):
        df = self._compute_metrics(folder, iou, ticks, func)
        self._plot_confusion_matrix(df.loc[conf])
        self._plot_pr_curve(df)
        self._plot_f1_curve(df)

    def _compute_metrics(self, folder, iou, ticks, func):
        dataset = YoloDataset(folder)
        conf_range = [round(x / ticks, 3) for x in range(ticks + 1)]
        df = pd.DataFrame(0, index=conf_range, columns=['tp', 'fp', 'fn'])

        for img, annotations in tqdm(dataset):
            predicted = self.predict(img, conf=0.0)[:, :-1]

            for i in conf_range:
                detections = predicted[predicted[:, 4] >= i][:, :-1]
                
                if len(detections) == 0:
                    df.loc[i, 'fn'] += len(annotations)
                elif len(annotations) == 0:
                    df.loc[i, 'fp'] += len(detections)
                else:
                    iou_matrix = func(annotations, detections)
                    best_iou, _ = iou_matrix.max(dim=1)
                    tp = (best_iou >= iou).sum().item()

                    df.loc[i, 'tp'] += tp
                    df.loc[i, 'fp'] += len(detections) - tp
                    df.loc[i, 'fn'] += len(annotations) - tp

        df['precision'] =  df['tp'] / (df['tp'] + df['fp'])
        df['precision'] = df['precision'].fillna(1)
        df['recall'] = df['tp'] / (df['tp'] + df['fn'])
        df['recall'] = df['recall'].fillna(0)
        df['f1_score'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])

        return df
    
    def plot(self, img, box: torch.Tensor):
        height, width, _ = img.shape

        box = box.tolist()
        x1, y1, x2, y2 = box
        x1 = int(round(x1 * width))
        x2 = int(round(x2 * width))
        y1 = int(round(y1 * height))
        y2 = int(round(y2 * height))

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=15)

        return img
    
    def _plot_confusion_matrix(self, df: pd.DataFrame):
        sns.heatmap([[df['tp'], df['fp']], [df['fn'], 0]], cmap='Blues', robust=True, annot=True, xticklabels=['gun', 'background'], yticklabels=['gun', 'background'], fmt='g')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

    def _plot_pr_curve(self, df):
        recall = [1] + df['recall'].tolist()
        precision = [0] + df['precision'].tolist()
        pr_curve = sns.lineplot(x=recall, y=precision, linewidth=2)

        fontsize = 22
        pr_curve.set_xlabel("Recall", fontsize=fontsize)
        pr_curve.set_ylabel("Precision", fontsize=fontsize)
        pr_curve.set_title("Precision-Recall Curve", fontsize=fontsize)

        pr_curve.set_xlim(0, 1)
        pr_curve.set_ylim(0, 1)

        pr_curve.legend(labels=[f'AUC = {auc(recall, precision):.2f}'], fontsize=fontsize-2)

        pr_curve.tick_params(axis='both', which='major', labelsize=fontsize-2)

        plt.show()
        return df
    
    def _plot_f1_curve(self, df):
        f1_curve = sns.lineplot(x=df.index.tolist(), y=df['f1_score'].tolist(), linewidth=2)

        fontsize = 22
        f1_curve.set_xlabel("Confidence", fontsize=fontsize)
        f1_curve.set_ylabel("F1", fontsize=fontsize)
        f1_curve.set_title("F1-Confidence Curve", fontsize=fontsize)

        f1_curve.set_xlim(0, 1)
        f1_curve.set_ylim(0, 1)

        f1_curve.legend(labels=[f'{df["f1_score"].max():.2f} at {df["f1_score"].idxmax():.2f}'], fontsize=fontsize-2)

        f1_curve.tick_params(axis='both', which='major', labelsize=fontsize-2)
        
        plt.show()
