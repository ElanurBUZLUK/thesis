# utils.py

import cv2
import numpy as np

def visualize_detections(image, detections, class_names):
    for detection in detections:
        label = detection['label']
        score = detection['score']
        bbox = detection['box']
        class_name = class_names[label]

        color = (0, 255, 0)  # Yeşil renk
        thickness = 2

        # Kareyi çiz
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Etiket ve skoru ekle
        text = f'{class_name}: {score:.2f}'
        cv2.putText(image, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image
