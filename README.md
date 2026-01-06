# Gözlük Çerçevesi Tespiti (Glasses Frame Detection)

Bu proje, **Digital Image Processing (CSE4xx)** dersi kapsamında geliştirilmiş olup, geleneksel görüntü işleme teknikleri kullanarak insan yüzündeki gözlük çerçevelerini tespit etmeyi ve işaretlemeyi amaçlar.

Derin öğrenme (Deep Learning) modelleri yerine, **OpenCV** kütüphanesi ile temel görüntü işleme algoritmaları (Kenar tespiti, Morfolojik işlemler, ROI analizi) kullanılarak algoritmik bir yaklaşım sergilenmiştir.

## 🎯 Proje Özellikleri

* **ROI Odaklı Analiz:** Tüm görüntü yerine Haar Cascade sınıflandırıcıları ile sadece yüz ve göz bölgesine odaklanarak işlem yükünü azaltır ve doğruluğu artırır.
* **Gradient Tabanlı Göz Bebeği Tespiti:** Timm-Barth yöntemine dayalı gradyan tabanlı algoritma ile değişken aydınlatma koşullarında güvenilir göz bebeği lokalizasyonu.
* **Çok Aşamalı Yeniden Deneme:** Normal, akıllı retry ve gevşek mod olmak üzere üç aşamalı strateji ile zorlu görüntülerde bile yüksek tespit oranı.
* **Adaptif Kenar Tespiti:** Canny algoritması ile görüntü istatistiklerine göre adaptif eşik değerleri.
* **Morfolojik İşlemler:** Kopuk çerçeve parçalarını birleştirmek için agresif morfolojik kapatma işlemleri.
* **Simetri Kontrolü:** Sol ve sağ göz için tespit edilen alanlar arasındaki simetri kontrolü ile dengeli tespit.
* **Görselleştirme:** Tespit edilen çerçeveleri orijinal görüntü üzerinde sarı renkte işaretler.

## 📊 Test Sonuçları

Sistem 22 farklı yüz görüntüsü üzerinde test edilmiştir:

* **Toplam Görüntü:** 22
* **Başarılı Tespit:** 19
* **Başarısız Tespit:** 3
* **Başarı Oranı:** **86.36%**

Başarılı tespitlerden 18 görüntüde 2 çerçeve, 1 görüntüde 1 çerçeve tespit edilmiştir.

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

Debug modu, işleme adımlarının ara görüntülerini `debug/` klasörüne kaydeder.

> **Not:** Sisteminizde OpenCV'nin dahili Haar cascade dosyaları yoksa,
> `haarcascades/haarcascade_frontalface_default.xml` ve
> `haarcascades/haarcascade_eye_tree_eyeglasses.xml` dosyalarını `haarcascades/` klasörüne
> indirmeniz gerekir. OpenCV'nin resmi GitHub deposundaki XML dosyaları
> kullanılabilir.

## 📂 Proje Yapısı

```text
glass/
│
├── src/
│   ├── main.py          # Ana uygulama kodu
│   └── archive/         # Eski versiyonlar
├── images/              # Test edilecek örnek fotoğraflar
├── output/              # İşlenmiş ve işaretlenmiş çıktıların kaydedildiği klasör
├── debug/               # Debug modunda ara adımlar (opsiyonel)
├── folder/
│   └── report.latex    # IEEE formatında akademik rapor
├── haarcascades/        # Haar Cascade XML model dosyaları
├── requirements.txt     # Gerekli kütüphane listesi
└── README.md            # Proje dokümantasyonu
```

## 🔗 GitHub Repository

Proje kaynak kodu ve dokümantasyon:

**GitHub Repository:** [https://github.com/ali-gurcan/eyeglassDetector](https://github.com/ali-gurcan/eyeglassDetector)

## 📝 Algoritma Akışı

1. **Yüz Tespiti:** Haar Cascade ile yüz bölgesi tespit edilir
2. **Göz Bölgesi Çıkarımı:** Yüzün üst %20-60 aralığından sol ve sağ göz bölgeleri çıkarılır
3. **Göz Bebeği Lokalizasyonu:** Gradient tabanlı yöntem ile göz bebeği merkezi bulunur
4. **Ön İşleme:** Bilateral filtre ve CLAHE ile kontrast artırılır
5. **Kenar Tespiti:** Adaptif Canny algoritması ile kenarlar bulunur
6. **Morfolojik İşlemler:** Kopuk parçalar birleştirilir
7. **Kontur Analizi:** RETR_TREE modu ile iç ve dış konturlar bulunur
8. **Filtreleme:** Alan, en-boy oranı, solidity ve konum filtreleri uygulanır
9. **Yeniden Deneme:** Gerekirse gevşek mod ve agresif parametrelerle tekrar denenir
10. **Sonuç:** Tespit edilen çerçeveler görüntü üzerine çizilir

## 🎓 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
