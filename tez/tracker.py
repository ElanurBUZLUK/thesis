# tracker.py

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from yolov9 import YOLOv9  # YOLOv9 modelini yükleyin veya kullanacağınız YOLO modelini buraya ekleyin

class TrackerYOLOv9:
    def __init__(self, model_cfg_path, model_weights_path, class_names):
        self.model = YOLOv9(model_cfg_path, num_classes=len(class_names))
        self.model.load_weights(model_weights_path)
        self.class_names = class_names
        self.tracks = []

    def detect_objects(self, image):
        # Giriş görüntüsünde nesne tespiti yap
        detections = self.model(image)
        return detections

    def update_tracks(self, image, detections):
        # Takip edilen nesneleri güncelle
        updated_tracks = []

        for detection in detections:
            label = detection['label']
            score = detection['score']
            bbox = detection['box']

            # Her bir tespiti bir takip nesnesi olarak temsil et
            new_track = {
                'label': label,
                'score': score,
                'bbox': bbox
            }

            updated_tracks.append(new_track)

        # Güncellenmiş takip listesini sakla
        self.tracks = updated_tracks

    def get_fish_count(self):
        return len(self.tracks)  # Takip edilen nesne sayısını balık sayısı olarak döndür
