# Eyeglass Lens Edge Detection — Labeled + Classical Image Processing Roadmap

## Goal
Optician’ın çektiği tek bir fotoğraftan `frame`/`glass` bölgelerini kullanarak **lens kenarını** mümkün olduğunca doğru şekilde tespit etmek.
Sistemi **tamamen on-device** (offline) hedefleyip “model + klasik image processing refinement” hibrit yaklaşımıyla kaliteyi artırmayı amaçlıyoruz.

## 1) Labeled Veri Hazırlığı (Dataset)
1. Etiketlenecek sınıflar: `frame`, `glass` (YOLOv8 segmentation uyumlu).
2. Her fotoğrafta:
   - Sol `frame` ve sağ `frame` polygonları (genelde kadraj sabitse daha hızlı olur).
   - Her `glass` için polygonlar (1/2 cam; aynı kadrajda tutarlı çizim hedeflenir).
3. Çıkış formatı: **YOLOv8 segmentation** (polygon/mask) export.

## 2) Eğitim / Model Tabanlı Tahmin
Amaç: Fotoğraftan tutarlı şekilde `frame` ve/veya `glass` mask/ROI elde etmek.

Seçenek A (YOLOv8 segmentation):
- `frame` ve/veya `glass` için YOLOv8 segmentation eğitimi.
- Çıktı: mask veya polygon → sonraki adım için ROI.

Seçenek B (Melez: Detector/ROI + Segmenter):
- Önce bir detector ile bölgeyi bulma (örn. `frame` için bbox/ROI).
- Sonra `MobileSAM` benzeri segmenter ile maske üretme.

## 3) Klasik Image Processing Refinement (Optician-grade edge)
Amaç: Modelin ürettiği mask/ROI içinde, **Canny + kontur hiyerarşisi** ile gerçek lens kenarını daha keskin ve doğru hale getirmek.

Önerilen refinement adımları:
1. ROI içinde preprocessing:
   - kontrast artırma (CLAHE vb.)
   - filtreleme (bilateral/gaussian vb.)
2. `Canny` ile kenar çıkarımı.
3. `cv2.findContours(..., cv2.RETR_TREE, ...)` ile hiyerarşiyi kullanma:
   - dış/parent kontur = frame çevresi
   - iç/child kontur = lens kenarı (RETR_TREE üzerinden)
4. Filtreleme (alan, aspekt oran, pozisyon vb.) ile yanlış konturları eleme.
5. Son konturu döndürüp ekrana overlay etme.

> Not: Bu yaklaşım kodda zaten “Canny + contour hierarchy (RETR_TREE) lens edge inside frame mask” mantığıyla kullanılıyor.

## 4) Hibrit Sistem Mantığı (Labeled + Classical Birlikte)
1. Labeled veri + model/segmenter: `frame` maskesi veya lens ROI üretir.
2. Klasik refinement: yalnızca bu ROI içinde çalışır.
3. Böylece:
   - yanlış edge’ler azalır
   - lens kenarı “mask sınırı” yerine gerçek kenara daha yakın elde edilir

## 5) Deney Planı (Karşılaştırma)
1. Baselines:
   - Sadece klasik CV (ROI yok / tüm görselde edge)
   - Sadece model/segmenter (refine yok)
2. Hibrit:
   - model/segmenter mask + Canny/RETR_TREE refinement
3. Ölçümler (öneri):
   - `frame`/`glass` mask kalitesi (IoU vb.)
   - lens edge kalitesi (IoU/mesafe tabanlı metrik veya görsel doğrulama)
   - süre/latency (on-device hedef)

## Deliverables
1. Etiketli dataset (YOLOv8 segmentation uyumlu).
2. En az bir trained model (YOLOv8-seg) veya segmenter tabanlı ROI üretim.
3. `refine_to_lens_edge` benzeri hibrit refinement pipeline çıktıları.
4. Baseline vs hibrit karşılaştırma görselleri ve kısa metrik özetleri.

## Optional: VLM Kullanımı (Auto-label / assisted labeling)
- VLM’i doğrudan final edge üretmek için değil; **ön-etiketleme/rough bbox** için kullanmak daha pratik.
- Sonrasında mask/polygon yine SAM/klasik refinement + manuel düzeltme ile finalize edilir.

