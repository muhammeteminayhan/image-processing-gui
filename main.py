import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QInputDialog, QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, \
    QVBoxLayout, QWidget, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ArithmeticWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görsel Aritmetik İşlemler")
        self.setGeometry(100, 100, 1200, 500)

        self.image1 = None
        self.image2 = None

        # Layout
        layout = QVBoxLayout()

        # Butonlar
        btn_layout = QHBoxLayout()
        self.load_btn1 = QPushButton("Görsel 1 Yükle")
        self.load_btn2 = QPushButton("Görsel 2 Yükle")
        self.process_btn = QPushButton("İşlem Yap")
        btn_layout.addWidget(self.load_btn1)
        btn_layout.addWidget(self.load_btn2)
        btn_layout.addWidget(self.process_btn)

        # Dropdown for choosing operation
        self.operation_dropdown = QComboBox()
        self.operation_dropdown.addItem("Çıkarma")
        self.operation_dropdown.addItem("Çarpma")
        self.operation_dropdown.addItem("Toplama")
        self.operation_dropdown.addItem("Bölme")
        btn_layout.addWidget(self.operation_dropdown)

        # Görsel gösterim alanları
        self.label1 = QLabel()
        self.label2 = QLabel()
        self.result_label = QLabel()

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.label1)
        img_layout.addWidget(self.label2)
        img_layout.addWidget(self.result_label)

        layout.addLayout(btn_layout)
        layout.addLayout(img_layout)
        self.setLayout(layout)

        # Bağlantılar
        self.load_btn1.clicked.connect(lambda: self.load_image(1))
        self.load_btn2.clicked.connect(lambda: self.load_image(2))
        self.process_btn.clicked.connect(self.process_images)

    def load_image(self, index):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            qimg = QImage(file_name)
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            width = qimg.width()
            height = qimg.height()

            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            img_data = np.array(ptr).reshape((height, width, 3))

            if index == 1:
                self.image1 = img_data.copy()
                self.label1.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio))
            else:
                self.image2 = img_data.copy()
                self.label2.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio))

    def process_images(self):
        if self.image1 is None or self.image2 is None:
            return

        # Seçilen işleme göre işlem yap
        operation = self.operation_dropdown.currentText()

        if operation == "Çıkarma":
            result = self.subtract_images(self.image1, self.image2)
        elif operation == "Çarpma":
            result = self.multiply_images(self.image1, self.image2)
        elif operation == "Toplama":
            result = self.add_images(self.image1, self.image2)
        elif operation == "Bölme":
            result = self.divide_images(self.image1, self.image2)

        # NumPy → QImage → QLabel
        height, width, _ = result.shape
        qimg = QImage(result.data, width, height, width * 3, QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio))

    def subtract_images(self, img1, img2):
        result = np.zeros_like(img1, dtype=np.uint8)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                for c in range(3):
                    val = int(img1[y, x, c]) - int(img2[y, x, c])
                    result[y, x, c] = max(0, min(255, val))
        return result

    def multiply_images(self, img1, img2):
        result = np.zeros_like(img1, dtype=np.uint8)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                for c in range(3):
                    val = int(img1[y, x, c]) * int(img2[y, x, c]) // 255  # Normalize for 8-bit image
                    result[y, x, c] = min(255, val)
        return result

    def add_images(self, img1, img2):
        result = np.zeros_like(img1, dtype=np.uint8)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                for c in range(3):
                    val = int(img1[y, x, c]) + int(img2[y, x, c])
                    result[y, x, c] = min(255, val)
        return result

    def divide_images(self, img1, img2):
        result = np.zeros_like(img1, dtype=np.uint8)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                for c in range(3):
                    val = int(img1[y, x, c]) // (int(img2[y, x, c]) + 1)  # Avoid division by zero
                    result[y, x, c] = min(255, val)
        return result






class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görüntü İşleme Projesi")
        self.setGeometry(0, 0, 1200, 830)
        self.image = None
        self.initUI()

    def initUI(self):
        self.label = QLabel("Görüntü Yüklenmedi")
        self.label.setFixedSize(500, 500)

        # Butonlar
        load_button = QPushButton("Görüntü Yükle")
        load_button.clicked.connect(self.load_image)

        clear_button = QPushButton("Temizle")
        clear_button.clicked.connect(self.clear_image)

        # Diğer butonlar
        gray_button = QPushButton("Gri Dönüşüm")
        gray_button.clicked.connect(self.convert_to_grayscale)

        binary_button = QPushButton("Binary Dönüşüm")
        binary_button.clicked.connect(self.binary_threshold)

        rotate_button = QPushButton("Görüntü Döndür")
        rotate_button.clicked.connect(self.rotate_image)

        crop_button = QPushButton("Görüntü Kırp")
        crop_button.clicked.connect(self.crop_image)

        zoom_button = QPushButton("Görüntü Yaklaştır/Uzaklaştır")
        zoom_button.clicked.connect(self.zoom_image)

        color_space_button = QPushButton("Renk Uzayı Dönüşümü")
        color_space_button.clicked.connect(self.convert_color_space)

        contrast_button = QPushButton("Kontrast Ayarla")
        contrast_button.clicked.connect(self.adjust_contrast)

        median_button = QPushButton("Medyan Filtresi")
        median_button.clicked.connect(self.apply_median_filter_button)

        double_threshold_button = QPushButton("Çift Eşikleme")
        double_threshold_button.clicked.connect(self.apply_double_threshold_button)

        edge_button = QPushButton("Kenar Bulma (Canny)")
        edge_button.clicked.connect(self.canny_edge_detection)

        # Gürültü Ekleme ve Filtreleme Butonu
        noise_filter_button = QPushButton("Gürültü Ekle ve Filtrele")
        noise_filter_button.clicked.connect(self.apply_noise_and_filter)

        motion_blur_button = QPushButton("Motion Blur Uygula")
        motion_blur_button.clicked.connect(self.apply_motion_blur)

        morphological_button = QPushButton("Morfolojik İşlemler Uygula")
        morphological_button.clicked.connect(self.apply_morphological_operations)

        self.btn_arithmetic = QPushButton("Görsel Aritmetik İşlemler")
        self.btn_arithmetic.clicked.connect(self.open_arithmetic_window)

        histogram_button = QPushButton("Histogram Germe")
        histogram_button.clicked.connect(self.histogram_equalization)

        self.histogram_canvas = PlotCanvas(self)

        # Yatay Layout
        layout_main = QHBoxLayout()

        # Sol taraf: Butonlar
        layout_buttons = QVBoxLayout()
        layout_buttons.addWidget(load_button)
        layout_buttons.addWidget(clear_button)  # Temizleme butonunu yükleme butonunun hemen altına ekliyoruz
        layout_buttons.addWidget(gray_button)
        layout_buttons.addWidget(binary_button)
        layout_buttons.addWidget(rotate_button)
        layout_buttons.addWidget(crop_button)
        layout_buttons.addWidget(zoom_button)
        layout_buttons.addWidget(color_space_button)
        layout_buttons.addWidget(contrast_button)
        layout_buttons.addWidget(median_button)
        layout_buttons.addWidget(double_threshold_button)
        layout_buttons.addWidget(edge_button)
        layout_buttons.addWidget(noise_filter_button)
        layout_buttons.addWidget(motion_blur_button)
        layout_buttons.addWidget(morphological_button)
        layout_buttons.addWidget(self.btn_arithmetic)
        layout_buttons.addWidget(histogram_button)
        layout_buttons.addWidget(self.histogram_canvas)

        # Sağ taraf: Resim
        layout_main.addLayout(layout_buttons)
        layout_main.addWidget(self.label)  # Resim gösterim alanı

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.image = self.imread(path)
            self.display_image(self.image)

    def clear_image(self):
        """
        Yüklenen resmi ve tüm işlenmiş verileri temizler.
        Görüntü sıfırlanır ve kullanıcıya temizlenmiş bir ekran gösterilir.
        """
        self.image = None  # Görüntüyü sıfırla
        self.label.setText("Görüntü Yüklenmedi")  # Ekranda "Görüntü Yüklenmedi" mesajı göster
        self.display_image(np.zeros((1, 1, 3), np.uint8))  # Ekranı temizle (boş bir görüntü ile)

    def imread(self, path):
        # Sadece RGB olarak oku
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def display_image(self, img_array):
        if img_array is None:
            return

        if len(img_array.shape) == 2:  # Gri tonluysa
            img_array = np.stack((img_array,) * 3, axis=-1)  # Gri -> RGB

        h, w, ch = img_array.shape
        bytes_per_line = ch * w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

    def convert_to_grayscale(self):
        if self.image is not None:
            gray = np.dot(self.image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            gray_rgb = np.stack((gray,)*3, axis=-1)
            self.display_image(gray_rgb)

    def convert_to_grayscale_array(self, image):
        """
        Görüntüyü gri tonlamaya çeviren fonksiyon.
        Bu versiyon, numpy array döndüren bir fonksiyon olacak.
        """
        if image is None:
            return None

        # Görüntüyü gri tonlamaya dönüştürmek için ağırlıklı ortalama yöntemi kullanıyoruz.
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return gray

    def binary_threshold(self):
        if self.image is not None:
            # Kullanıcıdan threshold değeri al
            thresh, ok = QInputDialog.getInt(self, "Eşik Değeri", "0-255 arası eşik değeri girin:", 128, 0, 255)
            if ok:
                gray = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                binary = np.where(gray > thresh, 255, 0).astype(np.uint8)
                binary_rgb = np.stack((binary,) * 3, axis=-1)
                self.display_image(binary_rgb)

    def binarize_image(self, gray_image, threshold_value=127):
        """
        Gri tonlamalı görüntüyü ikili görüntüye dönüştürür.
        threshold_value: Eşik değeri (default: 127).
        """
        binary_image = np.where(gray_image >= threshold_value, 255, 0).astype(np.uint8)
        return binary_image

    def rotate_image(self):
        if self.image is not None:
            angle, ok = QInputDialog.getDouble(self, "Açı Girin", "Görüntüyü kaç derece döndürmek istersiniz?", 90,
                                               -360, 360, 1)
            if ok:
                angle_rad = math.radians(angle)
                h, w, c = self.image.shape

                # Yeni boyutları hesapla
                new_h = int(abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad)))
                new_w = int(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad)))

                rotated = np.zeros((new_h, new_w, c), dtype=np.uint8)

                cx_old, cy_old = w // 2, h // 2
                cx_new, cy_new = new_w // 2, new_h // 2

                for y in range(new_h):
                    for x in range(new_w):
                        # Yeni merkezden eski merkeze göre koordinatlar
                        x_old = int((x - cx_new) * math.cos(-angle_rad) - (y - cy_new) * math.sin(-angle_rad) + cx_old)
                        y_old = int((x - cx_new) * math.sin(-angle_rad) + (y - cy_new) * math.cos(-angle_rad) + cy_old)

                        if 0 <= x_old < w and 0 <= y_old < h:
                            rotated[y, x] = self.image[y_old, x_old]

                self.display_image(rotated)

    def crop_image(self):
        if self.image is not None:
            h, w, _ = self.image.shape

            x, ok1 = QInputDialog.getInt(self, "X", f"Başlangıç X koordinatı (0 - {w - 1})", 0, 0, w - 1)
            if not ok1: return
            y, ok2 = QInputDialog.getInt(self, "Y", f"Başlangıç Y koordinatı (0 - {h - 1})", 0, 0, h - 1)
            if not ok2: return
            crop_w, ok3 = QInputDialog.getInt(self, "Genişlik", f"Kırpılacak genişlik (1 - {w - x})", w - x, 1, w - x)
            if not ok3: return
            crop_h, ok4 = QInputDialog.getInt(self, "Yükseklik", f"Kırpılacak yükseklik (1 - {h - y})", h - y, 1, h - y)
            if not ok4: return

            cropped = self.image[y:y + crop_h, x:x + crop_w]
            self.display_image(cropped)

    def zoom_image(self):
        if self.image is not None:
            scale, ok = QInputDialog.getDouble(self, "Yaklaştır/Uzaklaştır",
                                               "Ölçek oranı girin (örn. 2.0 = büyüt, 0.5 = küçült)", 1.0, 0.1, 10.0, 2)
            if ok:
                h, w, c = self.image.shape
                new_h, new_w = int(h * scale), int(w * scale)

                zoomed = np.zeros((new_h, new_w, c), dtype=np.uint8)

                for y in range(new_h):
                    for x in range(new_w):
                        src_x = min(int(x / scale), w - 1)
                        src_y = min(int(y / scale), h - 1)
                        zoomed[y, x] = self.image[src_y, src_x]

                self.display_image(zoomed)

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb / 255.0  # Normalize the RGB values
        c_max = max(r, g, b)
        c_min = min(r, g, b)
        delta = c_max - c_min

        if delta == 0:
            h = 0
        elif c_max == r:
            h = (60 * ((g - b) / delta)) % 360
        elif c_max == g:
            h = (60 * ((b - r) / delta)) + 120
        else:
            h = (60 * ((r - g) / delta)) + 240

        if c_max == 0:
            s = 0
        else:
            s = delta / c_max

        v = c_max

        return np.array([h, s * 100, v * 100])  # Return HSV in degrees and percentages

    def convert_color_space(self):
        if self.image is not None:
            conversion_type, ok = QInputDialog.getItem(self, "Renk Uzayı Dönüşümü", "Dönüşüm tipi seçin:",
                                                       ["RGB to HSV", "RGB to Grayscale"], 0, False)
            if ok:
                if conversion_type == "RGB to HSV":
                    hsv_image = np.zeros_like(self.image)
                    for i in range(self.image.shape[0]):
                        for j in range(self.image.shape[1]):
                            hsv_image[i, j] = self.rgb_to_hsv(self.image[i, j])
                    self.display_image(hsv_image)
                elif conversion_type == "RGB to Grayscale":
                    grayscale_image = np.dot(self.image[..., :3], [0.299, 0.587, 0.114])  # Convert RGB to grayscale
                    grayscale_image = np.stack((grayscale_image,) * 3, axis=-1)  # Convert back to 3 channels
                    self.display_image(grayscale_image.astype(np.uint8))

    def histogram_equalization(self):
        if self.image is not None:
            # 1. Grayscale görüntü oluşturma
            gray_image = np.dot(self.image[..., :3], [0.299, 0.587, 0.114])

            # 2. Kullanıcıdan min ve max değerlerini al
            min_val, ok1 = QInputDialog.getInt(self, "Histogram Parametreleri", "Minimum Değer Girin (0-255)", 0, 0,
                                               255, 1)
            max_val, ok2 = QInputDialog.getInt(self, "Histogram Parametreleri", "Maksimum Değer Girin (0-255)", 255, 0,
                                               255, 1)

            if ok1 and ok2:
                # 3. Histogram hesaplama ve normalizasyon
                hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
                cdf = hist.cumsum()  # Kümülatif dağılım fonksiyonu
                cdf_normalized = cdf * (max_val - min_val) / cdf[-1] + min_val  # Normalize et

                # 4. Histogram germe
                img_equalized = np.interp(gray_image.flatten(), bins[:-1], cdf_normalized)
                img_equalized = img_equalized.reshape(gray_image.shape)

                # 5. RGB'ye dönüşüm
                equalized_image = np.stack((img_equalized,) * 3, axis=-1)

                # 6. Histogramları çizme
                self.plot_histogram(gray_image, equalized_image)

                # 7. Görüntüleri gösterme
                self.display_image(equalized_image.astype(np.uint8))

    def plot_histogram(self, original_image, equalized_image):
        # 1. Orijinal ve eşitlenmiş histogramları hesapla
        hist_orig, bins_orig = np.histogram(original_image.flatten(), 256, [0, 256])
        hist_equalized, bins_equalized = np.histogram(equalized_image.flatten(), 256, [0, 256])

        # 2. Histogramları çiz
        self.histogram_canvas.axes.clear()
        self.histogram_canvas.axes.plot(bins_orig[:-1], hist_orig, color='blue', label='Orijinal Histogram')
        self.histogram_canvas.axes.plot(bins_equalized[:-1], hist_equalized, color='red', label='Eşitlenmiş Histogram')
        self.histogram_canvas.axes.legend()
        self.histogram_canvas.draw()

    def open_arithmetic_window(self):
        self.arithmetic_window = ArithmeticWindow()
        self.arithmetic_window.show()

    def adjust_contrast(self):
        if self.image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return

        contrast_value, ok = QInputDialog.getDouble(self, 'Kontrast Ayarla', 'Kontrast Değeri Girin (1.0 normal):', 1.0,
                                                    0.1, 5.0, 2)
        if ok:
            self.image = self.adjust_contrast_algorithm(self.image, contrast_value)
            self.display_image(self.image)

    def adjust_contrast_algorithm(self, image, contrast_value):
        # image bir NumPy array
        height, width = image.shape[:2]

        # Yeni görüntü için boş dizi oluştur
        new_image = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                if len(image.shape) == 3:  # Renkli görüntü
                    for c in range(3):  # BGR kanalları
                        new_pixel = image[y, x, c] * contrast_value
                        new_image[y, x, c] = max(0, min(255, int(new_pixel)))
                else:  # Gri tonlamalı görüntü
                    new_pixel = image[y, x] * contrast_value
                    new_image[y, x] = max(0, min(255, int(new_pixel)))

        return new_image

    def apply_median_filter(self, image, kernel_size=3):
        pad = kernel_size // 2
        height, width = image.shape[:2]
        new_image = np.zeros_like(image)

        # Görüntüyü kenarlardan sıfırlarla doldur
        if len(image.shape) == 3:
            padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        else:
            padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

        for y in range(height):
            for x in range(width):
                if len(image.shape) == 3:  # Renkli
                    for c in range(3):
                        window = padded_image[y:y + kernel_size, x:x + kernel_size, c].flatten()
                        median = np.median(window)
                        new_image[y, x, c] = int(median)
                else:  # Gri
                    window = padded_image[y:y + kernel_size, x:x + kernel_size].flatten()
                    median = np.median(window)
                    new_image[y, x] = int(median)

        return new_image

    def apply_median_filter_button(self):
        if self.image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return

        # Kullanıcıdan kernel boyutu al
        kernel_size, ok = QInputDialog.getInt(
            self, "Medyan Filtresi",
            "Kernel (Pencere) Boyutu Girin (tek sayı, örn: 3, 5, 7):",
            3, 3, 15, 2
        )

        if ok:
            if kernel_size % 2 == 0:
                QMessageBox.warning(self, "Hatalı Giriş", "Lütfen TEK sayı girin!")
                return

            self.image = self.apply_median_filter(self.image, kernel_size)
            self.display_image(self.image)

    def apply_double_threshold_button(self):
        if self.image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyin.")
            return

        t_low, ok1 = QInputDialog.getInt(self, "Alt Eşik", "Alt eşik değeri (0-255):", 50, 0, 255)
        if not ok1:
            return

        t_high, ok2 = QInputDialog.getInt(self, "Üst Eşik", "Üst eşik değeri (0-255):", 150, 0, 255)
        if not ok2:
            return

        if t_low >= t_high:
            QMessageBox.warning(self, "Hatalı Giriş", "Alt eşik üst eşikten küçük olmalıdır!")
            return

        mid_value, ok3 = QInputDialog.getInt(self, "Ara Değer", "Ara değer (0-255, genelde 127):", 127, 0, 255)
        if not ok3:
            return

        self.image = self.double_threshold(self.image, t_low, t_high, mid_value)
        self.display_image(self.image)

    def double_threshold(self, image, t_low, t_high, mid_value=127):
        if len(image.shape) == 3:
            # Griye çevir
            image = np.mean(image, axis=2).astype(np.uint8)

        height, width = image.shape
        new_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                if pixel >= t_high:
                    new_image[y, x] = 255
                elif pixel < t_low:
                    new_image[y, x] = 0
                else:
                    new_image[y, x] = mid_value

        return new_image

    def apply_gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        from math import exp, pi
        # 2D Gaussian kernel oluştur
        k = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for x in range(-k, k + 1):
            for y in range(-k, k + 1):
                kernel[x + k, y + k] = (1 / (2 * pi * sigma ** 2)) * exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        # padding
        padded = np.pad(image, ((k, k), (k, k)), mode='constant')
        filtered = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                filtered[i, j] = np.sum(region * kernel)

        return filtered.astype(np.uint8)

    def compute_gradients(self, image):
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        Ix = self.convolve(image, Kx)
        Iy = self.convolve(image, Ky)

        magnitude = np.hypot(Ix, Iy)
        magnitude = magnitude / magnitude.max() * 255

        direction = np.arctan2(Iy, Ix)
        return magnitude.astype(np.uint8), direction

    def non_maximum_suppression(self, magnitude, direction):
        Z = np.zeros_like(magnitude, dtype=np.uint8)
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                try:
                    q = 255
                    r = 255

                    # 0 derece
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = magnitude[i, j + 1]
                        r = magnitude[i, j - 1]
                    # 45 derece
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = magnitude[i + 1, j - 1]
                        r = magnitude[i - 1, j + 1]
                    # 90 derece
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = magnitude[i + 1, j]
                        r = magnitude[i - 1, j]
                    # 135 derece
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = magnitude[i - 1, j - 1]
                        r = magnitude[i + 1, j + 1]

                    if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                        Z[i, j] = magnitude[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        return Z

    def threshold_hysteresis(self, image, lowThreshold, highThreshold):
        M, N = image.shape
        res = np.zeros((M, N), dtype=np.uint8)

        weak = 75
        strong = 255

        strong_i, strong_j = np.where(image >= highThreshold)
        weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        # Hysteresis
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if res[i, j] == weak:
                    if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or
                            (res[i + 1, j + 1] == strong) or (res[i, j - 1] == strong) or
                            (res[i, j + 1] == strong) or (res[i - 1, j - 1] == strong) or
                            (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                        res[i, j] = strong
                    else:
                        res[i, j] = 0
        return res

    def canny_edge_detection(self):
        if self.image is None:
            return

        gray = np.dot(self.image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        ksize, ok1 = QInputDialog.getInt(self, "Gaussian Çekirdek Boyutu", "Tek sayı girin (örn. 3, 5, 7):", 5)
        sigma, ok2 = QInputDialog.getDouble(self, "Sigma Değeri", "Gaussian için sigma değeri:", 1.0)
        low_thres, ok3 = QInputDialog.getInt(self, "Düşük Eşik", "0-255 arası:", 50)
        high_thres, ok4 = QInputDialog.getInt(self, "Yüksek Eşik", "0-255 arası:", 150)

        if not (ok1 and ok2 and ok3 and ok4): return

        blurred = self.apply_gaussian_filter(gray, ksize, sigma)
        mag, dir = self.compute_gradients(blurred)
        suppressed = self.non_maximum_suppression(mag, dir)
        final_edges = self.threshold_hysteresis(suppressed, low_thres, high_thres)

        edge_rgb = np.stack((final_edges,) * 3, axis=-1)
        self.display_image(edge_rgb)

    def convolve(self, image, kernel):
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        output = np.zeros_like(image)

        for i in range(img_h):
            for j in range(img_w):
                region = padded_image[i:i + k_h, j:j + k_w]
                output[i, j] = np.sum(region * kernel)

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    def add_salt_and_pepper_noise(self, image, noise_ratio):
        row, col, ch = image.shape
        s_vs_p = 0.5  # Salt ve pepper oranı (0.5 %50 salt, %50 pepper)
        amount = noise_ratio  # Gürültü oranı

        # Salt (beyaz) ekleme
        num_salt = int(amount * image.size * s_vs_p)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        image[salt_coords[0], salt_coords[1], :] = 255

        # Pepper (siyah) ekleme
        num_pepper = int(amount * image.size * (1. - s_vs_p))
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        image[pepper_coords[0], pepper_coords[1], :] = 0

        return image

    def mean_filter(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        return self.convolve(image, kernel)

    def median_filter(self, image, kernel_size):
        pad = kernel_size // 2
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        output = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.median(region, axis=(0, 1))

        return output

    def apply_noise_and_filter(self):
        if self.image is None:
            return

        # 1. Kullanıcıdan gürültü oranını al
        noise_ratio, ok1 = QInputDialog.getDouble(self, "Gürültü Oranı", "Gürültü oranını girin (0-1 arası):", 0.01, 0,
                                                  1)
        if not ok1:
            return

        # 2. Görüntüye gürültü ekle
        noisy_image = self.add_salt_and_pepper_noise(self.image.copy(), noise_ratio)

        # 3. Filtreleme türünü al
        filter_type, ok2 = QInputDialog.getItem(self, "Filtre Türü", "Filtre türünü seçin:", ["Mean", "Median"], 0,
                                                False)
        if not ok2:
            return

        # 4. Filtre boyutunu al
        kernel_size, ok3 = QInputDialog.getInt(self, "Filtre Boyutu", "Filtre boyutunu girin (tek sayı):", 3, 1, 15)
        if not ok3:
            return

        # 5. Filtreyi uygula
        if filter_type == "Mean":
            filtered_image = self.mean_filter(noisy_image, kernel_size)
        elif filter_type == "Median":
            filtered_image = self.median_filter(noisy_image, kernel_size)

        # 6. Sonucu görüntüle
        self.display_image(filtered_image)

    def apply_motion_blur(self):
        if self.image is None:
            return

        # Kullanıcıdan kernel boyutunu al
        kernel_size, ok1 = QInputDialog.getInt(self, "Kernel Boyutu", "Kernel boyutunu girin (tek sayı):", 5, 1, 15)
        if not ok1:
            return

        # Kullanıcıdan hareket yönünü al
        direction, ok2 = QInputDialog.getItem(self, "Hareket Yönü", "Hareket yönünü seçin:",
                                              ["Yatay", "Dikey", "Özel Açılı"], 0, False)
        if not ok2:
            return

        if direction == "Yatay":
            kernel = self.create_motion_kernel(kernel_size, "horizontal")
        elif direction == "Dikey":
            kernel = self.create_motion_kernel(kernel_size, "vertical")
        else:
            # Kullanıcıdan açı girilsin
            angle, ok3 = QInputDialog.getInt(self, "Açı", "Hareket açısını girin (0-180):", 45, 0, 180)
            if not ok3:
                return
            kernel = self.create_motion_kernel(kernel_size, angle)

        # Uygulanan kernel ile konvolüsyon işlemi
        blurred_image = self.apply_convolution(self.image, kernel)

        self.image = blurred_image
        self.display_image(blurred_image)

    def create_motion_kernel(self, size, direction_or_angle):
        # Hareket yönüne veya açıya göre kernel oluşturma
        kernel = np.zeros((size, size))

        if direction_or_angle == "horizontal":
            kernel[int((size - 1) / 2), :] = np.ones(size)  # Yatay hareket
        elif direction_or_angle == "vertical":
            kernel[:, int((size - 1) / 2)] = np.ones(size)  # Dikey hareket
        else:
            # Açıya göre kernel oluşturma
            angle = np.deg2rad(direction_or_angle)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dx = int(np.cos(angle) * (i - center) - np.sin(angle) * (j - center))
                    dy = int(np.sin(angle) * (i - center) + np.cos(angle) * (j - center))
                    if 0 <= dx + center < size and 0 <= dy + center < size:
                        kernel[dx + center, dy + center] = 1

        # Kernel toplamını normalize et (toplam 1 olsun)
        return kernel / np.sum(kernel)

    def apply_convolution(self, image, kernel):
        # Görüntüye konvolüsyon işlemi yapan fonksiyon
        height, width, channels = image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Görüntüye padding ekle
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant',
                              constant_values=0)

        # Çıktı görüntüsünü oluştur
        output = np.zeros_like(image)

        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    # Kernel'i görüntü üzerinde kaydır
                    region = padded_image[i:i + kernel_height, j:j + kernel_width, k]
                    output[i, j, k] = np.sum(region * kernel)

        return output.astype(np.uint8)

    def dilation(self, binary_image, kernel):
        """
        Genişleme (Dilation) işlemi
        """
        image_height, image_width = binary_image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Görüntüye padding ekliyoruz
        padded_image = np.pad(binary_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)

        dilated_image = np.zeros_like(binary_image)

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                dilated_image[i, j] = np.max(region * kernel)

        return dilated_image

    def erosion(self, binary_image, kernel):
        """
        Aşınma (Erosion) işlemi
        """
        image_height, image_width = binary_image.shape
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Görüntüye padding ekliyoruz
        padded_image = np.pad(binary_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)

        eroded_image = np.zeros_like(binary_image)

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                eroded_image[i, j] = np.min(region * kernel)

        return eroded_image

    def opening(self, binary_image, kernel):
        """
        Açma (Opening) işlemi: Aşınma ardından genişleme
        """
        eroded = self.erosion(binary_image, kernel)
        opened_image = self.dilation(eroded, kernel)
        return opened_image

    def closing(self, binary_image, kernel):
        """
        Kapama (Closing) işlemi: Genişleme ardından aşınma
        """
        dilated = self.dilation(binary_image, kernel)
        closed_image = self.erosion(dilated, kernel)
        return closed_image

    def apply_morphological_operations(self):
        if self.image is None:
            return

        # Kullanıcıdan işlem türünü al
        operation, ok1 = QInputDialog.getItem(self, "İşlem Türü", "İşlem türünü seçin:",
                                              ["Genişleme", "Aşınma", "Açma", "Kapama"], 0, False)
        if not ok1:
            return

        # Kullanıcıdan kernel boyutunu al
        kernel_size, ok2 = QInputDialog.getInt(self, "Kernel Boyutu", "Kernel boyutunu girin (tek sayı):", 3, 1, 15)
        if not ok2:
            return

        # Gri tonlama (binary image) öncesi kontrol
        gray = self.convert_to_grayscale_array(self.image)  # Gri tonlama işlemi
        binary_image = self.binarize_image(gray, threshold_value=127)  # İkili (binary) dönüşüm

        # Kernel oluşturma
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Seçilen işleme göre fonksiyon çağrılır
        if operation == "Genişleme":
            result_image = self.dilation(binary_image, kernel)
        elif operation == "Aşınma":
            result_image = self.erosion(binary_image, kernel)
        elif operation == "Açma":
            result_image = self.opening(binary_image, kernel)
        elif operation == "Kapama":
            result_image = self.closing(binary_image, kernel)

        self.image = result_image
        self.display_image(result_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())