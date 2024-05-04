# test_yolov9.py

from dataset_custom import CustomDataset
from yolov9 import YOLOv9
import torch
from torch.utils.data import DataLoader

def test_yolov9_model(data_dir, model_cfg_path, model_weights_path, class_names):
    # Veri Kümesi ve DataLoader oluştur
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Modeli yükle
    model = YOLOv9(model_cfg_path, num_classes=len(class_names))
    model.load_weights(model_weights_path)
    model.eval()

    # Test Döngüsü
    fish_count_total = 0
    for images, targets in dataloader:
        # Giriş görüntülerini al ve modeli kullanarak nesne tespiti yap
        outputs = model(images)
        # Tespitleri işle ve balık sayısını hesapla
        fish_count = ...  # Tespitlerden balık sayısını al
        fish_count_total += fish_count

    # Toplam balık sayısını yazdır
    print(f"Toplam balık sayısı test setinde: {fish_count_total}")

    return fish_count_total
