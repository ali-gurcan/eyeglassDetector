# Gözlük Çerçevesi Tespiti (Glasses Frame Detection)

Bu proje, **Digital Image Processing (CSE4xx)** dersi kapsamında geliştirilmiş olup, geleneksel görüntü işleme teknikleri kullanarak insan yüzündeki gözlük çerçevelerini tespit etmeyi ve işaretlemeyi amaçlar.

Derin öğrenme (Deep Learning) modelleri yerine, **OpenCV** kütüphanesi ile temel görüntü işleme algoritmaları (Kenar tespiti, Morfolojik işlemler, ROI analizi) kullanılarak algoritmik bir yaklaşım sergilenmiştir.

## 🎯 Proje Özellikleri

* **ROI Odaklı Analiz:** Tüm görüntü yerine Haar Cascade sınıflandırıcıları ile sadece yüz ve göz bölgesine odaklanarak işlem yükünü azaltır ve doğruluğu artırır.
* **Canlı Parametre Ayarı (Trackbars):** Işık koşullarına göre `Canny Edge Threshold` ve `Dilation` değerlerini gerçek zamanlı ayarlamak için kullanıcı arayüzü sunar.
* **Gürültü Filtreleme:** Gaussian Blur ve Kontur Alanı (Contour Area) filtreleri ile çerçeve dışındaki detayları (kirpik, yansıma vb.) eler.
* **Görselleştirme:** Tespit edilen çerçeveleri orijinal görüntü üzerinde renklendirerek gösterir.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Kütüphaneler:**
    * `OpenCV (cv2)`: Görüntü işleme algoritmaları için.
    * `NumPy`: Matris işlemleri için.

## 🚀 Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Çalıştırma

### Batch Modu (Tüm Resimleri İşle)

Varsayılan çalışma şekli batch modudur; `images/` klasöründeki tüm resimleri işler ve sonuçları `output/` klasörüne kaydeder:

```bash
# Virtual environment ile
source .venv/bin/activate
python src/main.py

# Debug modu ile (ara adımları görmek için)
python src/main.py --debug
```

### 🎨 İnteraktif GUI Modu (Renk Seçici)

Modern Tkinter tabanlı GUI ile çerçeve rengini tıklayarak seçebilirsiniz:

```bash
# Sistem Python ile (Tkinter desteği için)
/usr/bin/python3 src/gui.py

# Veya direkt GUI dosyasını çalıştır
/usr/bin/python3 src/gui.py
```

**GUI Kullanımı:**
1. "Resim Yükle" butonuna tıklayın
2. Gözlük çerçevesine tıklayarak rengi seçin
3. Tolerance slider ile hassasiyeti ayarlayın
4. "Sonucu Kaydet" ile sonucu kaydedin

> **Not:** Homebrew Python'unda Tkinter yoksa sistem Python'unu (`/usr/bin/python3`) kullanın. Paketler otomatik olarak yüklenecektir.

> Not: Sisteminizde OpenCV'nin dahili Haar cascade dosyaları yoksa,
> `haarcascades/haarcascade_frontalface_default.xml` ve
> `haarcascades/haarcascade_eye.xml` dosyalarını `haarcascades/` klasörüne
> indirmeniz gerekir. OpenCV'nin resmi GitHub deposundaki XML dosyaları
> kullanılabilir.

## 📂 Proje Yapısı

```text
glasses-detection-project/
│
├── src/
│   └── main.py          # Ana uygulama kodu
├── images/              # Test edilecek örnek fotoğraflar
├── output/              # İşlenmiş ve işaretlenmiş çıktıların kaydedildiği klasör
├── haarcascades/        # (Opsiyonel) .xml model dosyaları
├── requirements.txt     # Gerekli kütüphane listesi
└── README.md            # Proje dokümantasyonu
