# Colab Pro - V4 Modeli Eğitim Yol Haritası 🚀

Bu doküman, Google Colab Pro üzerinde yepyeni ve birleştirilmiş devasa V4 veri setimizi (orijinal + sentetik sahte parlamalı resimler) eğitmek için kullanacağın Python kodlarını ve adımları içerir. 

## 📌 Adım 1: Veri Setini Yükleme
Hazırlayacağımız `v4_dataset.zip` dosyasını Google Drive'ına yükle. (Örneğin `MyDrive/v4_dataset.zip` olarak).

## 📌 Adım 2: Colab Pro'yu Hazırlama
1. Google Colab'de yeni bir Not Defteri (Notebook) aç.
2. **Çalışma Zamanı (Runtime) -> Çalışma zamanı türünü değiştir** menüsünden **T4 GPU**, **L4 GPU** veya **A100 GPU** seç (Colab Pro abonesi olduğun için L4 veya A100 seçmen harika olur).

## 📌 Adım 3: Hücre Hücre Çalıştırılacak Kodlar

Aşağıdaki kodları Colab'de sırasıyla yeni hücrelere yapıştır ve çalıştır.

### Hücre 1: Google Drive'a Bağlanma ve Dosyaları Çıkarma
```python
from google.colab import drive
import zipfile
import os

# 1. Google Drive'a bağlan
drive.mount('/content/drive')

# 2. V4 Veri Setini SSD'ye (Colab içine) çıkar
zip_path = '/content/drive/MyDrive/v4_dataset.zip' # Eğer farklı bir klasördeyse burayı değiştir
extract_path = '/content/v4_dataset'

print("Veri seti çıkarılıyor, lütfen bekleyin...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("✅ Veri seti başarıyla çıkarıldı!")
```

### Hücre 2: YOLO Kütüphanesini Kurma
```python
!pip install ultralytics -q
import ultralytics
ultralytics.checks()
```

### Hücre 3: YOLO11 ile V4 Modelini Eğitme (The Beast Mode)
> **Not:** iPhone 16 için en iyi performans/kalite dengesini veren model `yolo11m-seg.pt` (Medium) modelidir. Eğer en üst sınırları zorlamak istersen `yolo11x-seg.pt` (XLarge) yapabilirsin.

```python
from ultralytics import YOLO

# En yeni YOLO11 Medium modelini başlat
model = YOLO('yolo11m-seg.pt')

# Eğitimi başlat (1024 piksel çözünürlük ile en ince cam kenarları için)
results = model.train(
    data='/content/v4_dataset/data.yaml', # Veri seti konfigürasyon dosyası
    epochs=200,                           # Maksimum tur sayısı
    patience=50,                          # 50 tur boyunca iyileşme olmazsa erken durdur
    imgsz=1024,                           # Mükemmel kalite için yüksek çözünürlük
    batch=-1,                             # GPU RAM'ine göre otomatik ayarla (AutoBatch)
    project='iphone16_glass_v4',          # Klasör adı
    name='weights',                       # Alt klasör adı
    device=0                              # GPU'yu kullan
)
```

### Hücre 4: iPhone 16 İçin Apple CoreML'e Çevirme ve Kaydetme
Eğitim bittikten sonra, modelin beyni olan `best.pt` dosyasını doğrudan iPhone 16'nın (Xcode) anlayacağı `best.mlpackage` formatına çevireceğiz ve Drive'a yedekleyeceğiz.

```python
import shutil

# 1. Eğitilen modeli CoreML formatına çevir (iPhone 16 Neural Engine için NPU uyumlu)
print("Apple CoreML formatına çevriliyor...")
exported_model = model.export(format='coreml', nms=True, imgsz=1024)
print("✅ Çeviri başarılı!")

# 2. Üretilen dosyaları Google Drive'a kalıcı olarak kaydet (Yedekleme)
drive_yedek_klasoru = '/content/drive/MyDrive/iPhone16_V4_Model'
os.makedirs(drive_yedek_klasoru, exist_ok=True)

shutil.copy('/content/iphone16_glass_v4/weights/weights/best.pt', f'{drive_yedek_klasoru}/best_v4.pt')
shutil.copy('/content/iphone16_glass_v4/weights/weights/best.mlpackage', f'{drive_yedek_klasoru}/best_v4.mlpackage')

print(f"🎉 İşlem tamamlandı! Modelin Drive'daki şu klasöre kopyalandı: {drive_yedek_klasoru}")
```

---
Bu adımları tamamladığında, elinde her türlü parlama ve kötü koşula dayanıklı, doğrudan iPhone'un beynine gömülmeye hazır son teknoloji bir CoreML paketi olacak!
