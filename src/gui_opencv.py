#!/usr/bin/env python3
"""
OpenCV-based GUI for Eyeglass Frame Detection
Simple, reliable interface with color picker using OpenCV's built-in GUI
"""

import cv2
import numpy as np
import os
from pathlib import Path
import glob

# Tkinter'ı sadece dosya seçici için kullan
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# --- Ayarlar ---
DEFAULT_IMAGE_FOLDER = 'images'
OUTPUT_FOLDER = 'output'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class OpenCVColorPicker:
    def __init__(self):
        self.cv_image = None
        self.processed_image = None
        self.target_hsv = None
        self.tolerance = 40
        self.original_path = None
        
        # Mouse callback için
        self.window_name = "Gözlük Çerçevesi Sihirbazı - Çerçeveye Tıkla!"
        self.current_mask = None
        
        # Tıklanan nokta bilgisi (görselleştirme için)
        self.clicked_point = None  # (x, y) tuple
        self.clicked_color_bgr = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse tıklama olayını yakala"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.cv_image is not None:
                h, w = self.cv_image.shape[:2]
                
                # Görüntü boyutunu hesapla (display için)
                display_h = 400
                display_w = int(w * display_h / h)
                
                # Sadece sol panelde (orijinal resim) tıklama geçerli
                if 0 <= x < display_w and 0 <= y < display_h:
                    # Tıklanan koordinatı orijinal resme çevir
                    orig_x = int(x * w / display_w)
                    orig_y = int(y * h / display_h)
                    
                    # Sınırları kontrol et
                    if 0 <= orig_x < w and 0 <= orig_y < h:
                        # Tıklanan pikselin rengini al
                        clicked_bgr = self.cv_image[orig_y, orig_x]
                        clicked_hsv_pixel = cv2.cvtColor(
                            np.uint8([[clicked_bgr]]), cv2.COLOR_BGR2HSV
                        )[0][0]
                        self.target_hsv = clicked_hsv_pixel
                        self.clicked_point = (orig_x, orig_y)
                        self.clicked_color_bgr = clicked_bgr
                        print(f"✅ Renk Seçildi: HSV {self.target_hsv} | BGR {clicked_bgr} | Konum: ({orig_x}, {orig_y})")
                        self.update_processing()
    
    def tolerance_callback(self, val):
        """Tolerance trackbar callback"""
        self.tolerance = val
        if self.target_hsv is not None:
            self.update_processing()
    
    def update_processing(self):
        """Renk seçimine göre işlemi güncelle"""
        if self.cv_image is None or self.target_hsv is None:
            return
        
        hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        
        # HSV sınırları
        lower_bound = np.array([
            max(0, self.target_hsv[0] - self.tolerance),
            max(0, self.target_hsv[1] - self.tolerance),
            max(0, self.target_hsv[2] - self.tolerance)
        ])
        
        upper_bound = np.array([
            min(179, self.target_hsv[0] + self.tolerance),
            min(255, self.target_hsv[1] + self.tolerance),
            min(255, self.target_hsv[2] + self.tolerance)
        ])
        
        # Maske oluştur
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Morfolojik temizlik
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sonuç resmini hazırla
        result_img = self.cv_image.copy()
        img_area = result_img.shape[0] * result_img.shape[1]
        
        # Konturları çiz
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > img_area * 0.001:  # Çok küçük noktaları çizme
                hull = cv2.convexHull(cnt)
                cv2.drawContours(result_img, [hull], -1, (0, 255, 255), 2)  # Sarı
        
        self.processed_image = result_img
        self.current_mask = mask
        
        # Tüm görüntüleri tek bir pencerede birleştir (yan yana)
        self.show_combined_view()
    
    def show_combined_view(self):
        """Tüm görüntüleri tek bir pencerede göster"""
        if self.cv_image is None:
            return
        
        h, w = self.cv_image.shape[:2]
        
        # Görüntüleri aynı boyuta getir
        display_h = 400  # Sabit yükseklik
        display_w = int(w * display_h / h)  # Oranı koru
        
        # Orijinal resmi yeniden boyutlandır
        orig_display = cv2.resize(self.cv_image, (display_w, display_h)).copy()
        
        # Tıklanan noktayı görsel olarak işaretle (Color Picker efekti)
        if self.clicked_point is not None:
            click_x, click_y = self.clicked_point
            # Display koordinatlarına çevir
            disp_x = int(click_x * display_w / w)
            disp_y = int(click_y * display_h / h)
            
            # Büyük daire (dış halka)
            cv2.circle(orig_display, (disp_x, disp_y), 20, (0, 255, 0), 2)
            # Küçük daire (iç nokta)
            cv2.circle(orig_display, (disp_x, disp_y), 3, (0, 255, 0), -1)
            # Çapraz çizgiler (crosshair)
            cv2.line(orig_display, (disp_x - 15, disp_y), (disp_x + 15, disp_y), (0, 255, 0), 1)
            cv2.line(orig_display, (disp_x, disp_y - 15), (disp_x, disp_y + 15), (0, 255, 0), 1)
            
            # Seçilen rengi göster (küçük renk kutusu)
            if self.clicked_color_bgr is not None:
                color_box_size = 30
                color_box_y = max(50, disp_y - 40)
                color_box_x = disp_x - color_box_size // 2
                if color_box_x < 0:
                    color_box_x = disp_x + 25
                cv2.rectangle(orig_display, 
                             (color_box_x, color_box_y), 
                             (color_box_x + color_box_size, color_box_y + color_box_size),
                             tuple(map(int, self.clicked_color_bgr)), -1)
                cv2.rectangle(orig_display, 
                             (color_box_x, color_box_y), 
                             (color_box_x + color_box_size, color_box_y + color_box_size),
                             (255, 255, 255), 2)
        
        # Maske varsa göster, yoksa siyah
        if self.current_mask is not None:
            # Maske üzerine renkli overlay ekle (daha görsel)
            mask_colored = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
            # Maske alanlarını vurgula
            mask_display = cv2.resize(mask_colored, (display_w, display_h))
            # Maske alanlarını yeşil tonlarda göster
            mask_green = mask_display.copy()
            mask_green[:, :, 0] = 0  # Blue channel'ı sıfırla
            mask_green[:, :, 2] = 0  # Red channel'ı sıfırla
            # Orijinal resimle blend yap
            mask_overlay = cv2.addWeighted(
                cv2.resize(self.cv_image, (display_w, display_h)), 0.6,
                mask_green, 0.4, 0
            )
            mask_display = mask_overlay
        else:
            mask_display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            cv2.putText(mask_display, "Renk Seçilmedi", (10, display_h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Sonuç varsa göster, yoksa orijinal
        if self.processed_image is not None:
            result_display = cv2.resize(self.processed_image, (display_w, display_h))
        else:
            result_display = orig_display.copy()
        
        # Etiketler ekle
        cv2.putText(orig_display, "Orijinal (Tikla)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(mask_display, "Maske (Overlay)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_display, "Sonuc", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Tolerance bilgisini göster
        if self.target_hsv is not None:
            info_text = f"Tolerance: {self.tolerance}"
            cv2.putText(orig_display, info_text, (10, display_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Yan yana birleştir
        combined = np.hstack([orig_display, mask_display, result_display])
        
        # Pencereyi göster
        cv2.imshow(self.window_name, combined)
    
    def load_image(self, image_path):
        """Resim yükle"""
        self.original_path = image_path
        self.cv_image = cv2.imread(image_path)
        
        if self.cv_image is None:
            print(f"❌ Resim okunamadı: {image_path}")
            return False
        
        # Çok büyükse küçült
        h, w = self.cv_image.shape[:2]
        max_width = 1200
        if w > max_width:
            scale = max_width / w
            self.cv_image = cv2.resize(self.cv_image, None, fx=scale, fy=scale)
        
        self.processed_image = self.cv_image.copy()
        self.target_hsv = None  # Seçimi sıfırla
        self.current_mask = None  # Maskeyi sıfırla
        self.clicked_point = None  # Tıklanan noktayı sıfırla
        self.clicked_color_bgr = None  # Seçilen rengi sıfırla
        
        # İlk görüntüyü göster
        self.show_combined_view()
        
        return True
    
    def select_image_file(self):
        """Dosya seçici ile resim seç"""
        if not HAS_TKINTER:
            print("❌ Dosya seçici için Tkinter gerekli!")
            print("   Alternatif: Resmi 'images/' klasörüne koyun ve 'N'/'P' tuşlarını kullanın.")
            return None
        
        # Tkinter root penceresini gizle
        root = tk.Tk()
        root.withdraw()  # Ana pencereyi gizle
        root.attributes('-topmost', True)  # Üste getir
        
        # Dosya seçici
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_IMAGE_FOLDER,
            title="Resim Seç",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("WebP", "*.webp"),
                ("All Files", "*.*")
            ]
        )
        
        root.destroy()  # Tkinter penceresini kapat
        
        return file_path if file_path else None
    
    def save_result(self):
        """Sonucu kaydet"""
        if self.processed_image is None or self.original_path is None:
            print("❌ Kaydedilecek bir işlem yok!")
            return
        
        filename = os.path.basename(self.original_path)
        save_path = os.path.join(OUTPUT_FOLDER, f"clicked_{filename}")
        cv2.imwrite(save_path, self.processed_image)
        print(f"✅ Kaydedildi: {save_path}")
    
    def run(self, initial_image_path=None):
        """Ana döngü
        
        Args:
            initial_image_path: Başlangıçta yüklenecek resim yolu (opsiyonel)
        """
        # Tek pencere oluştur
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 500)  # Geniş pencere
        
        # Mouse callback (sadece sol panelde çalışacak)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Trackbar oluştur
        cv2.createTrackbar(
            'Tolerance',
            self.window_name,
            self.tolerance,
            100,
            self.tolerance_callback
        )
        
        # Resim listesi (images/ klasöründen)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(glob.glob(os.path.join(DEFAULT_IMAGE_FOLDER, ext)))
        
        current_idx = -1  # -1 = dosya seçici kullanılacak
        
        # Başlangıçta dosya seçici göster
        print("\n" + "="*60)
        print("🎨 Gözlük Çerçevesi Sihirbazı")
        print("="*60)
        print("📌 Kullanım:")
        print("  1. Resimdeki gözlük çerçevesine TIKLA (renk seç)")
        print("  2. Tolerance trackbar ile hassasiyeti ayarla")
        print("  3. 'O' tuşu: Resim seç (dosya seçici)")
        print("  4. 'S' tuşu: Sonucu kaydet")
        print("  5. 'N' tuşu: Sonraki resim (images/ klasöründen)")
        print("  6. 'P' tuşu: Önceki resim (images/ klasöründen)")
        print("  7. 'Q' veya ESC: Çıkış")
        print("="*60 + "\n")
        
        # İlk resmi seç
        if initial_image_path:
            # Komut satırından verilen resim
            if not self.load_image(initial_image_path):
                cv2.destroyAllWindows()
                return
            print(f"📷 Yüklenen: {os.path.basename(initial_image_path)}")
        elif image_files:
            # Önce dosya seçici göster
            selected_file = self.select_image_file()
            if selected_file:
                if not self.load_image(selected_file):
                    cv2.destroyAllWindows()
                    return
                print(f"📷 Yüklenen: {os.path.basename(selected_file)}")
            else:
                # Dosya seçilmediyse images/ klasöründen ilk resmi yükle
                current_idx = 0
                if not self.load_image(image_files[current_idx]):
                    cv2.destroyAllWindows()
                    return
                print(f"📷 Resim: {os.path.basename(image_files[current_idx])}")
        else:
            # images/ klasörü boşsa sadece dosya seçici
            selected_file = self.select_image_file()
            if not selected_file:
                print("❌ Resim seçilmedi. Çıkılıyor...")
                cv2.destroyAllWindows()
                return
            if not self.load_image(selected_file):
                cv2.destroyAllWindows()
                return
            print(f"📷 Yüklenen: {os.path.basename(selected_file)}")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q veya ESC
                break
            elif key == ord('o'):  # Open - Dosya seçici
                selected_file = self.select_image_file()
                if selected_file:
                    if self.load_image(selected_file):
                        print(f"📷 Yüklenen: {os.path.basename(selected_file)}")
                        # Dosya seçildiyse images/ klasöründeki index'i sıfırla
                        current_idx = -1
            elif key == ord('s'):  # Save
                self.save_result()
            elif key == ord('n'):  # Next
                if image_files and len(image_files) > 0:
                    if current_idx == -1:
                        current_idx = 0
                    else:
                        current_idx = (current_idx + 1) % len(image_files)
                    if self.load_image(image_files[current_idx]):
                        print(f"📷 Resim: {os.path.basename(image_files[current_idx])}")
                else:
                    print("❌ 'images/' klasöründe resim yok. 'O' tuşu ile resim seçin.")
            elif key == ord('p'):  # Previous
                if image_files and len(image_files) > 0:
                    if current_idx == -1:
                        current_idx = len(image_files) - 1
                    else:
                        current_idx = (current_idx - 1) % len(image_files)
                    if self.load_image(image_files[current_idx]):
                        print(f"📷 Resim: {os.path.basename(image_files[current_idx])}")
                else:
                    print("❌ 'images/' klasöründe resim yok. 'O' tuşu ile resim seçin.")
        
        cv2.destroyAllWindows()
        print("\n👋 Çıkılıyor...")


if __name__ == "__main__":
    import sys
    app = OpenCVColorPicker()
    
    # Komut satırından resim yolu al
    initial_image = None
    if len(sys.argv) > 1:
        initial_image = sys.argv[1]
        if not os.path.exists(initial_image):
            print(f"❌ Resim bulunamadı: {initial_image}")
            initial_image = None
    
    app.run(initial_image_path=initial_image)

