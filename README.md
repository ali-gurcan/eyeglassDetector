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

Varsayılan çalışma şekli başsızdır; tek komutla görüntüyü işler ve sonucu
dosyaya kaydeder:

```bash
python src/main.py \
  --image images/IMG-6151046557543560876\ copy.png \
  --output output/IMG-6151046557543560876_copy_detected.png
```

İsterseniz eşikleri komut satırından da düzenleyebilirsiniz:

```bash
python src/main.py --image images/pm0571_m0.jpg \
  --canny-min 30 --canny-max 160 \
  --dilation 2 --min-area 80
```

Trackbar arayüzünü açmak için `--interactive` bayrağını eklemeniz yeterlidir.
Bu modda aşağıdaki ayarlar pencereden kontrol edilir:

- `Canny Min` / `Canny Max`: Kenar tespit eşikleri
- `Dilation`: Kenarları kalınlaştırma iterasyonu
- `Min Area`: Kontur alan filtreleme eşiği

Trackbar modundan çıkmak için `q` tuşuna basmanız gerekir.

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
