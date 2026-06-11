# Gözlük Camı Tespiti (V4) - Başarı Yol Haritası 🚀

Bu yol haritası, bugüne kadar başardığımız yüksek kaliteli sonuçları alıp **üretim kalitesine (production grade)** taşımak ve iPhone 16 entegrasyonunu tamamlamak için yarın izleyeceğimiz adımları içerir.

## 🎯 Hedef
Elmas keskinliğinde, farklı ışık ve parlama koşullarından etkilenmeyen, doğrudan iPhone 16 Neural Engine üzerinde çevrimdışı ve saniyenin onda biri hızında çalışacak "Kusursuz V4 Modeli"ni oluşturmak.

---

## 1. Veri Seti Harmanlama (Data Merging)
Elimizde şu an orijinal resimler ve onlara eklediğimiz zorlayıcı (glare, blur, color) sentetik resimler var.
- Orijinal `images/` klasörü ile `manipulated_images/` klasörünü birleştirip toplam **894 fotoğraflık devasa bir görsel havuzu** oluşturacağız.
- Orijinal `labels/` ile bugün kopyaladığımız `manipulated_labels/` klasörünü birleştirerek etiket havuzunu tamamlayacağız.

## 2. Eğitim ve Doğrulama Ayırımı (Train/Val Split)
- Birleştirdiğimiz bu 894 resimlik dev veri setini rastgele olarak **%80 Eğitim (Train)** ve **%20 Test (Validation)** olacak şekilde ayıracağız.
- Yeni `data.yaml` dosyamızı oluşturup bu yepyeni ve yenilmez veri setini `v4_dataset.zip` olarak paketleyeceğiz.

## 3. Google Colab Pro ile V4 Eğitimi (The Beast)
- Yeni zip dosyasını Google Drive'ına yükleyeceksin.
- Bugün yaptığımız gibi Colab Pro'yu (L4 veya A100 GPU) açacağız.
- Bu kez **YOLO11 Medium** veya sınırları zorlamak istersen **YOLO11 XLarge** modelini kullanarak, 1024 piksel çözünürlükte en az 200 epoch sürecek nihai eğitimi başlatacağız.
- Model, veri setindeki sahte parlamaları ve bozuklukları görerek her türlü kötü şarta bağışıklık kazanacak.

## 4. Apple CoreML Export ve iOS Entegrasyonu
- Eğitim biter bitmez oluşan `best.pt` dosyasını `best.mlpackage` formatına çevireceğiz.
- Bu CoreML paketini indirip doğrudan iPhone 16 Xcode projene sürükleyip bırakacağız.
- Vision kütüphanesi (veya CoreML API) kullanarak kameradan gelen görüntüyü saniyede 30 kare hızında (real-time) çevrimdışı işleyecek Swift kodlarını hazırlayacağız.

---

> **Yarınki İlk İşimiz:** "Harmanlama ve Paketleme" işlemi olacak. Sen hazır olduğunda "Veri setini birleştirelim" demen yeterli. Arka planda saniyeler içinde tüm klasörleri birleştirip zipleyecek kodu çalıştıracağım.
