#!/usr/bin/env python3
"""
Tkinter-based GUI for Eyeglass Frame Detection
Modern, user-friendly interface with color picker
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path

# --- Ayarlar ---
DEFAULT_IMAGE_FOLDER = 'images'
OUTPUT_FOLDER = 'output'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class EyeglassDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gözlük Çerçevesi Sihirbazı (Renk Seçici)")
        self.root.geometry("1200x800")

        # Değişkenler
        self.cv_image = None  # Orijinal OpenCV resmi
        self.display_image = None  # Ekranda gösterilen resim
        self.processed_image = None  # İşlenmiş sonuç
        self.mask = None  # Renk maskesi
        self.original_path = None

        # Seçilen Renk (HSV)
        self.target_hsv = None

        # GUI Elemanları
        self.create_widgets()

    def create_widgets(self):
        # Üst Panel (Butonlar)
        top_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        top_frame.pack(fill=tk.X)

        btn_load = tk.Button(
            top_frame,
            text="Resim Yükle",
            command=self.load_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        btn_load.pack(side=tk.LEFT, padx=20)

        self.lbl_instruction = tk.Label(
            top_frame,
            text="Lütfen bir resim yükleyin...",
            bg="#f0f0f0",
            font=("Arial", 12)
        )
        self.lbl_instruction.pack(side=tk.LEFT, padx=20)

        btn_save = tk.Button(
            top_frame,
            text="Sonucu Kaydet",
            command=self.save_result,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        btn_save.pack(side=tk.RIGHT, padx=20)

        # Orta Panel (Resim Alanı)
        self.canvas_frame = tk.Frame(self.root, bg="#333")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # Tıklama Olayı

        # Alt Panel (Ayarlar)
        bottom_frame = tk.Frame(self.root, bg="#e0e0e0", pady=10)
        bottom_frame.pack(fill=tk.X)

        tk.Label(
            bottom_frame,
            text="Renk Toleransı:",
            bg="#e0e0e0",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=10)

        self.slider_tolerance = tk.Scale(
            bottom_frame,
            from_=5,
            to=100,
            orient=tk.HORIZONTAL,
            length=300,
            command=self.update_processing
        )
        self.slider_tolerance.set(40)  # Varsayılan tolerans
        self.slider_tolerance.pack(side=tk.LEFT, padx=10)

        tk.Label(
            bottom_frame,
            text="(Tıkladıktan sonra hassasiyeti buradan ayarla)",
            bg="#e0e0e0",
            font=("Arial", 9, "italic")
        ).pack(side=tk.LEFT)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=DEFAULT_IMAGE_FOLDER,
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.webp")]
        )
        if not file_path:
            return

        self.original_path = file_path
        self.cv_image = cv2.imread(file_path)

        if self.cv_image is None:
            messagebox.showerror("Hata", "Resim okunamadı!")
            return

        # Resmi ekrana sığacak kadar küçült (Gerekirse)
        h, w = self.cv_image.shape[:2]
        if w > 1000:
            scale = 1000 / w
            self.cv_image = cv2.resize(self.cv_image, None, fx=scale, fy=scale)

        self.processed_image = self.cv_image.copy()
        self.show_image(self.cv_image)
        self.lbl_instruction.config(
            text="Şimdi resimdeki GÖZLÜK ÇERÇEVESİNE tıkla!",
            fg="red"
        )
        self.target_hsv = None  # Önceki seçimi sıfırla

    def show_image(self, cv_img):
        # OpenCV (BGR) -> PIL (RGB)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Canvas boyutunu al
        self.canvas.update()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Resmi canvas'a sığdır
        img_w, img_h = pil_img.size
        scale_w = cw / img_w if cw > 0 else 1
        scale_h = ch / img_h if ch > 0 else 1
        scale = min(scale_w, scale_h, 1.0)  # Büyütme yapma, sadece küçült

        if scale < 1.0:
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil_img)

        # Canvas'ı temizle ve resmi ortala
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img, anchor=tk.CENTER)

        # Store scale for click coordinates
        self.display_scale = scale
        self.display_offset_x = (cw - pil_img.width) // 2
        self.display_offset_y = (ch - pil_img.height) // 2

    def on_canvas_click(self, event):
        if self.cv_image is None:
            return

        # Tıklanan koordinatları bul (Canvas ortalaması hesaba katılmalı)
        img_x = int((event.x - self.display_offset_x) / self.display_scale)
        img_y = int((event.y - self.display_offset_y) / self.display_scale)

        # Resim sınırları içinde mi?
        h, w = self.cv_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            # Tıklanan pikselin rengini al (BGR)
            clicked_bgr = self.cv_image[img_y, img_x]

            # HSV'ye çevir (Renk takibi için daha iyidir)
            clicked_hsv_pixel = cv2.cvtColor(
                np.uint8([[clicked_bgr]]), cv2.COLOR_BGR2HSV
            )[0][0]
            self.target_hsv = clicked_hsv_pixel

            print(f"Seçilen Renk (HSV): {self.target_hsv}")
            self.lbl_instruction.config(
                text=f"Renk Seçildi: HSV{self.target_hsv}. Tolerans çubuğu ile oyna.",
                fg="green"
            )

            # İşlemi başlat
            self.process_frame_by_color()

    def update_processing(self, val):
        if self.target_hsv is not None:
            self.process_frame_by_color()

    def process_frame_by_color(self):
        if self.cv_image is None or self.target_hsv is None:
            return

        hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        tolerance = int(self.slider_tolerance.get())

        # Alt ve Üst Sınırları Belirle
        # Hue (Renk Özü) daireseldir (0-179), taşmaları kontrol etmeliyiz ama basitlik için clip kullanıyoruz.
        lower_bound = np.array([
            max(0, self.target_hsv[0] - tolerance),
            max(0, self.target_hsv[1] - tolerance),
            max(0, self.target_hsv[2] - tolerance)
        ])

        upper_bound = np.array([
            min(179, self.target_hsv[0] + tolerance),
            min(255, self.target_hsv[1] + tolerance),
            min(255, self.target_hsv[2] + tolerance)
        ])

        # 1. Maske Oluştur (Seçilen renge uyan pikseller Beyaz, diğerleri Siyah)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 2. Morfolojik Temizlik (Gürültüleri At)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Küçük noktaları sil
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Boşlukları doldur

        # 3. Konturları Bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sonuç resmini hazırla
        result_img = self.cv_image.copy()

        # 4. Sadece mantıklı boyuttaki konturları çiz
        img_area = result_img.shape[0] * result_img.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > img_area * 0.001:  # Çok çok küçük noktaları çizme
                # Daha düzgün görünüm için Convex Hull
                hull = cv2.convexHull(cnt)
                cv2.drawContours(result_img, [hull], -1, (0, 255, 255), 2)  # Sarı Çerçeve

        self.processed_image = result_img
        self.show_image(self.processed_image)

    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek bir işlem yok!")
            return

        if self.original_path is None:
            messagebox.showwarning("Uyarı", "Önce bir resim yükleyin!")
            return

        filename = os.path.basename(self.original_path)
        save_path = os.path.join(OUTPUT_FOLDER, f"clicked_{filename}")
        cv2.imwrite(save_path, self.processed_image)
        messagebox.showinfo("Başarılı", f"Resim kaydedildi:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EyeglassDetectorApp(root)
    root.mainloop()

