# Git.py Dosyası - Detaylı Açıklama

## Genel Bakış

git.py dosyası, görüntülerde gözlük çerçevesi tespiti yapan bir Python scriptidir. Dosyanın temel felsefesi tespit oranını öncelemektir. Yani estetikten önce çerçeveyi bulmak hedeflenir. Bu nedenle dosya adı "Aggressive Eyeglass Detection" olarak belirtilmiştir.

## Dosya Yapısı

### Import Edilen Kütüphaneler

Script başında gerekli kütüphaneler import edilir:
- `glob`: Dosya arama için
- `os`: İşletim sistemi işlemleri için
- `shutil`: Dosya ve klasör işlemleri için
- `pathlib`: Dosya yolu işlemleri için
- `cv2`: OpenCV görüntü işleme kütüphanesi
- `numpy`: Sayısal hesaplamalar için

### Yapılandırma Değişkenleri

- `INPUT_FOLDER = 'images'`: Giriş görüntülerinin bulunduğu klasör
- `OUTPUT_FOLDER = 'output'`: Çıkış görüntülerinin kaydedileceği klasör
- `FACE_CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'`: Yüz tespiti için kullanılacak Haar Cascade modelinin yolu

Script çalıştığında output klasörü otomatik olarak oluşturulur.

## Fonksiyonlar

### 1. clear_output_dir()

Bu fonksiyon output klasörünü temizler. Klasör yoksa oluşturur, varsa içindeki tüm dosya ve klasörleri siler. Hata durumunda sessizce devam eder.

**İşlevi:**
- Output klasörünü oluşturur (yoksa)
- Klasör içindeki tüm dosyaları siler
- Klasör içindeki tüm alt klasörleri siler
- Hata durumunda sessizce devam eder

### 2. load_face_cascade()

Bu fonksiyon yüz tespiti için Haar Cascade sınıflandırıcısını yükler. Önce belirtilen yolda dosyayı arar, bulamazsa OpenCV'nin varsayılan haarcascades klasöründen yüklemeyi dener.

**İşlevi:**
- Belirtilen yolda Haar Cascade dosyasını arar
- Bulamazsa OpenCV'nin varsayılan klasöründen yükler
- CascadeClassifier nesnesini döndürür

### 3. find_frame_candidate(roi_gray, sensitivity='normal')

Bu fonksiyon çerçeve tespitinin ana mantığını içerir. İki parametre alır: roi_gray (gri tonlu göz bölgesi) ve sensitivity (normal veya relaxed). Fonksiyon dört ana adımda çalışır.

#### Adım 1: Ön İşleme

**Bilateral Filtre:**
- Gürültü azaltma için bilateral filtre uygulanır
- Kenarları korurken gürültüyü atar
- Parametreler: pencere boyutu 9, renk uzayı 75, uzamsal uzaklık 75

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
- Kontrast artırma için CLAHE uygulanır
- Normal modda clip limit 2.0, relaxed modda 4.0
- Tile grid boyutu 8x8
- Bu işlem koyu çerçeveleri belirginleştirir

#### Adım 2: Kenar Tespiti (Canny)

- Canny algoritması kullanılır
- Eşikler görüntünün medyan değerine göre adaptif hesaplanır
- Normal modda sigma 0.33, relaxed modda 0.50
- Alt eşik: medyan değerinin %67'si
- Üst eşik: medyan değerinin %150'si
- Bu sayede zayıf kenarlar da yakalanır

#### Adım 3: Morfolojik Kapatma

- Kırık çerçeve parçaları birleştirilir
- Normal modda 3x3, relaxed modda 5x5 eliptik kernel kullanılır
- İki iterasyon uygulanır
- Boşluklar doldurulur

#### Adım 4: Kontur Bulma

- RETR_TREE modu kullanılır
- İç içe konturlar da bulunur
- Böylece dış çerçeve kaybolsa bile iç çerçeve sınırları tespit edilebilir
- CHAIN_APPROX_SIMPLE ile kontur noktaları sadeleştirilir

#### Adım 5: Filtreleme

Her kontur için üç filtre uygulanır:

**Alan Filtresi:**
- Konturun alanı ROI'nin belirli bir yüzdesi içinde olmalı
- Normal modda: %5 ile %60 arası
- Relaxed modda: %2 ile %80 arası

**En-Boy Oranı Filtresi:**
- Konturun genişliği ve yüksekliği arasındaki oran kontrol edilir
- Normal modda: 0.8 ile 4.0 arası
- Relaxed modda: 0.5 ile 6.0 arası

**Konum Filtresi (Kaş Kontrolü):**
- ROI'nin en üst %5'inde olan konturlar kaş olarak değerlendirilip elenir

#### Adım 6: En İyi Aday Seçimi

- Filtrelerden geçen tüm konturlar alanlarına göre sıralanır
- En büyük alanlı kontur seçilir
- Eğer hiç aday yoksa None döndürülür

### 4. process_single_eye_roi(roi_gray, roi_color, offset_x, offset_y)

Bu fonksiyon tek bir göz bölgesini işler. Dört parametre alır: roi_gray (gri tonlu göz bölgesi), roi_color (renkli görüntü), offset_x ve offset_y (koordinat ofset değerleri).

**İşleyiş:**
1. ROI'nin boş olup olmadığı kontrol edilir
2. Plan A: Normal hassasiyetle çerçeve aranır
3. Plan B: Bulunamazsa relaxed mod denenir
4. Çerçeve bulunursa convex hull ile şekil düzeltilir
5. Kontur koordinatlarına offset eklenerek görüntü koordinatlarına taşınır
6. Cyan renkte, kalınlık 2 ile çizilir
7. Başarılı tespit durumunda 1, aksi halde 0 döndürülür

**Convex Hull:**
- Kırık parçaları birleştirip daha düzgün bir şekil oluşturur
- "Lastik bant" etkisi yaratır

### 5. process_image(img_path, face_cascade)

Bu fonksiyon tek bir görüntüyü işler. İki parametre alır: img_path (görüntü dosyası yolu) ve face_cascade (yüz tespiti sınıflandırıcısı).

**İşleyiş:**
1. Görüntü okunur
2. Görüntü genişliği 1000 pikselden büyükse ölçeklenir
3. Görüntü gri tonluya çevrilir
4. Haar Cascade ile yüzler tespit edilir
5. Yüz bulunamazsa fallback mekanizması devreye girer
6. Birden fazla yüz varsa en büyük yüz seçilir
7. Her yüz için göz bölgeleri belirlenir
8. Her göz bölgesi için çerçeve tespiti yapılır
9. Sonuç görüntü output klasörüne kaydedilir
10. Konsola bilgi mesajı yazdırılır

**Göz Bölgesi Belirleme:**
- Yüzün üst %18'inden %55'ine kadar olan kısım göz bölgesi olarak alınır
- Yüz dikey olarak ikiye bölünerek sol ve sağ göz bölgeleri oluşturulur

**Fallback Mekanizması:**
- Yüz bulunamazsa görüntünün ortasına yakın varsayılan bir bölge kullanılır
- Bu sayede yüz tespiti başarısız olsa bile çerçeve tespiti yapılabilir

### 6. main()

Bu fonksiyon programın ana giriş noktasıdır.

**İşleyiş:**
1. Başlangıç mesajı yazdırılır
2. Yüz tespiti sınıflandırıcısı yüklenir
3. Output klasörü temizlenir
4. Images klasöründeki tüm görüntü dosyaları bulunur
5. Her görüntü için process_image fonksiyonu çağrılır
6. İşlem bitince bitiş mesajı yazdırılır

**Desteklenen Formatlar:**
- JPG
- JPEG
- PNG
- WEBP

## Algoritma Akışı

```
Görüntü
  ↓
Yüz Tespiti (Haar Cascade)
  ↓
Göz Bölgeleri Belirleme (Sol/Sağ ROI)
  ↓
Her ROI için:
  ├─ Bilateral Filtre (Gürültü Azaltma)
  ├─ CLAHE (Kontrast Artırma)
  ├─ Canny Edge Detection (Kenar Tespiti)
  ├─ Morfolojik Kapatma (Boşluk Doldurma)
  ├─ Kontur Bulma (RETR_TREE)
  ├─ Filtreleme (Alan, En-Boy, Konum)
  ├─ En Büyük Aday Seçimi
  └─ Retry (Normal → Gevşek)
  ↓
Convex Hull (Şekil Düzeltme)
  ↓
Çizim (Cyan Renk)
```

## Önemli Tasarım Kararları

1. **Tespit Oranı Öncelikli:** Estetikten önce çerçeveyi bulmak hedeflenir
2. **İki Aşamalı Retry:** Normal mod başarısız olursa relaxed mod denenir
3. **Morfolojik Kapatma:** Kırık çerçeveleri birleştirir
4. **RETR_TREE:** İç çerçeve sınırlarını da bulur
5. **Adaptif Parametreler:** Hassasiyet moduna göre değişir

## Kullanım

Script çalıştırıldığında:
1. Images klasöründeki tüm görüntüler işlenir
2. Sonuçlar output klasörüne kaydedilir
3. Dosya adlarına "_aggressive" eki eklenir
4. Konsola işlem sonuçları yazdırılır

## Çıktı Formatı

- Başarılı tespit: `[BASARILI] dosya_adi.png -> 2 çerçeve bulundu.`
- Başarısız tespit: `[UYARI] dosya_adi.png -> Tüm denemelere rağmen bulunamadı.`

## Sonuç

Bu algoritma, zorlu görüntülerde bile gözlük çerçevesi tespit edebilmek için tasarlanmıştır. İki aşamalı retry mekanizması, morfolojik kapatma ve toleranslı filtreler sayesinde yüksek tespit oranı sağlar.

