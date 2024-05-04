# main.py

from dataset_custom import CustomDataset
from train_yolov9 import train_yolov9_model
from test_yolov9 import test_yolov9_model
from tracker import TrackerYOLOv9
import torch

def main():
    # Veri dizini
    train_data_dir = 'custom_train_dataset'
    test_data_dir = 'custom_test_dataset'

    # YOLOv9 modeli için eğitim
    model_cfg_path = 'yolov9.cfg'
    model_weights_path = 'yolov9.weights'
    class_names = ['fish']  # Özelleştirilmiş sınıf isimleri
    trained_model = train_yolov9_model(train_data_dir, model_cfg_path, model_weights_path, class_names)

    # YOLOv9 modeliyle nesne tespiti ve takip için tracker oluştur
    tracker = TrackerYOLOv9(model_cfg_path, model_weights_path, class_names)

    # Test seti üzerinde YOLOv9 ile nesne tespiti ve takip
    fish_count_total = test_yolov9_model(test_data_dir, model_cfg_path, model_weights_path, class_names)
    print(f"Toplam balık sayısı: {fish_count_total}")

if __name__ == '__main__':
    main()
