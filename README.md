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

### Yerel Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Docker ile Kurulum

Docker kullanarak projeyi çalıştırmak için:

```bash
# Docker image oluştur
docker build -t eyeglass-detector .

# Container çalıştır
docker run -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output eyeglass-detector
```

Veya docker-compose kullanarak:

```bash
docker-compose up
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
│   └── main.py          # Ana uygulama kodu
├── images/              # Test edilecek örnek fotoğraflar
├── output/              # İşlenmiş ve işaretlenmiş çıktıların kaydedildiği klasör
├── haarcascades/        # Haar Cascade XML model dosyaları
├── requirements.txt     # Gerekli kütüphane listesi
├── Dockerfile           # Docker image tanımı
├── docker-compose.yml   # Docker Compose konfigürasyonu
├── .dockerignore        # Docker ignore dosyası
└── README.md            # Proje dokümantasyonu
```

## 🔗 GitHub Repository

Proje kaynak kodu ve dokümantasyon:

**GitHub Repository:** [https://github.com/ali-gurcan/eyeglassDetector](https://github.com/ali-gurcan/eyeglassDetector)

## 📝 Algoritma Akışı

### Genel Akış

```
main() 
  ↓
1. Cascade'leri yükle (Face + Eye)
  ↓
2. Output klasörünü temizle
  ↓
3. images/ klasöründeki tüm resimleri bul
  ↓
4. Her resim için: process_image()
```

### process_image() Akışı

#### Aşama 1: Yüz Tespiti
```
1. Resmi oku ve boyutlandır (>1000px ise küçült)
2. Gray scale'e çevir
3. Haar Cascade ile yüz tespiti
4. Yüz yoksa fallback: varsayılan bölge
5. En büyük yüzü seç
```

#### Aşama 2: Göz Bölgelerini Ayır
```
Her yüz için:
  - Göz şeridi hesapla (yüzün %20-60 arası)
  - Sol ve sağ ROI'leri ayır (ortadan böl)
  - Her ROI için: process_single_eye_roi()
```

#### Aşama 3: Symmetry Check & Retry
```
Her yüz için:
  - Sol ve sağ göz sonuçlarını karşılaştır
  - Radius farkı > %40 ise:
    → Eksik/küçük göz için Guided Retry
    → Başarılı gözün radius'unu kullan
    → forced_iters=2, guided_radius=başarılı_radius
```

#### Aşama 4: Sonuçları Çiz ve Kaydet
```
- Bulunan konturları resme çiz (sarı renk)
- output/ klasörüne kaydet
```

### process_single_eye_roi() Akışı

#### Ön Hazırlık
```
1. Pupil center hesapla (get_precise_pupil_center)
   - Timm-Barth gradient-based yöntemi
   - ROI'nin merkez %40'ında ara
```

#### PLAN A: Normal Mode
```
find_frame_candidate(sensitivity='normal', morph_iters=1)
  ↓
Başarılı mı?
  ├─ EVET → Alan kontrolü (%10'dan küçükse at)
  └─ HAYIR → Smart Retry
```

#### Smart Retry (Plan A Başarısızsa)
```
Koşul: best_cnt None VEYA alan < %15
  ↓
find_frame_candidate(sensitivity='relaxed', morph_iters=3, use_mask=True)
  - Daha agresif birleştirme
  - Pupil bölgesini blurla (gözü gizle)
```

#### PLAN B: Relaxed Mode
```
Koşul: Plan A + Smart Retry başarısız
  ↓
find_frame_candidate(sensitivity='relaxed', morph_iters=2)
  - Daha toleranslı filtreler
  - Alan kontrolü (%10'dan küçükse at)
```

#### PLAN C: Metallic Mode
```
Koşul: Plan A + B başarısız
  ↓
find_frame_candidate(sensitivity='metallic', morph_iters=2, inflate=True)
  - Adaptive thresholding (Green channel)
  - Geometric completion
  - Radial edge scan
  - Inflate aktif (%15-25 genişletme)
```

### find_frame_candidate() Akışı

#### 1. Ön İşleme Stratejisi

**Metallic Mode:**
```
1. Green channel kullan (BGR → G)
2. Bilateral filter (hafif blur)
3. CLAHE (clipLimit=3.0)
4. Adaptive Threshold (block=15, C=3)
5. Median blur (gürültü temizleme)
```

**Normal/Relaxed Mode:**
```
1. Bilateral filter
2. CLAHE (clipLimit: normal=2.0, relaxed=4.0)
3. Pupil detection (get_precise_pupil_center)
4. Optional: Dynamic pupil blurring (use_mask=True)
5. Canny edge detection
```

#### 2. Morphological Closing
```
Kernel boyutu:
  - Normal: (3,3)
  - Relaxed: (5,5)
  - Metallic: (3,3)

Iterations: morph_iters (1-3 arası)
```

#### 3. Kontur Bulma
```
cv2.findContours(RETR_TREE)
  - İç içe konturları bul (hierarchy)
  - İç kontur = cam deliği (öncelikli)
```

#### 4. Kontur Filtreleme
```
Her kontur için:
  ├─ Alan kontrolü (min: %5-10, max: %30-50)
  ├─ Aspect ratio (0.4-2.2 veya 0.4-3.0)
  ├─ Solidity (0.70-0.85)
  ├─ Konum (kaş kontrolü: y < %15)
  └─ Merkezden uzaklık (max: %45)
```

#### 5. Shape Completion
```
Her geçerli kontur için:
  - FitEllipse ile elips uydur
  - 360 derece kapalı çokgen oluştur
  - Fallback: ConvexHull
```

#### 6. Scoring
```
Score = area × bottomness²
  - bottomness: Alt konum ağırlığı (y_max/h_roi)²
  - İç kontur ise: score × 3.0 (öncelik)
```

#### 7. Geometric Completion (Sadece Metallic)
```
1. İlk 10 adaydan, pupil'e mesafesi uygun olanları topla
   (est_radius × 0.6 < dist < est_radius × 2.0)
2. Yeterli nokta yoksa (<50):
   → Radial edge scan (72 ray, her 5 derece)
3. Tüm noktaları birleştir
4. ConvexHull ile çerçeveyi tamamla
```

#### 8. Final İşlemler
```
1. ConvexHull uygula
2. Inflate (metallic mode'da):
   - Normal: %15 genişlet
   - Küçükse: %25 genişlet
3. Tightening (daraltma):
   - Erode (morph_iters // 2)
   - Şişmeyi geri al
```

### Yardımcı Fonksiyonlar

#### get_precise_pupil_center()
- **Timm-Barth gradient-based yöntemi**
- Gradient vektörlerinin birleştiği noktayı bul
- Işık değişimlerine dayanıklı

#### radial_edge_scan()
- Pupil merkezinden 72 ray (her 5°)
- Her ray'de gradient magnitude'u yüksek noktayı bul
- Guided mode: dar arama (0.8x-1.2x)
- Normal mode: geniş arama (0.6x-1.4x)

#### scale_contour()
- Konturu merkez etrafında ölçeklendir
- Inflate için kullanılır

### Özet: 3 Mod Sistemi

1. **Normal:** Sıkı filtreler, düşük iterasyon
2. **Relaxed:** Toleranslı filtreler, orta iterasyon
3. **Metallic:** Adaptive threshold, geometric completion, inflate

Her mod başarısız olursa bir sonrakine geçer. Symmetry check ile başarılı gözün bilgisi diğer göze aktarılır.

## 🎓 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
