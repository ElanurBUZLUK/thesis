# train_yolov9.py

from dataset_custom import CustomDataset
from yolov9 import YOLOv9
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_yolov9_model(data_dir, model_cfg_path, model_weights_path, class_names):
    # Veri Kümesi ve DataLoader oluştur
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Modeli yükle
    model = YOLOv9(model_cfg_path, num_classes=len(class_names))
    model.load_weights(model_weights_path)
    model.train()

    # Eğitim Parametreleri
    criterion = ...  # Loss fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Eğitim Döngüsü
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            # Giriş görüntülerini al ve modeli eğit
            outputs = model(images)
            # Loss hesapla ve geri yayılım
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Eğitilmiş modeli döndür
    return model
