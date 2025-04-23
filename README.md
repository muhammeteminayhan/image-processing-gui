# 🖼️ Görüntü İşleme Arayüzü (Python - PyQt5)

Bu projede, PyQt5 ile geliştirilmiş kullanıcı arayüzü üzerinden çalışan bir görüntü işleme uygulaması yer almaktadır. Proje kapsamında, **hazır kütüphane fonksiyonları kullanılmadan**, temel görüntü işleme algoritmaları sıfırdan yazılmıştır.

---

## 📌 Özellikler

Bu uygulama üzerinden yapılabilecek işlemler şunlardır:

- ✅ Gri Dönüşüm  
- ✅ Binary Dönüşüm  
- ✅ Görüntü Döndürme  
- ✅ Görüntü Kırpma  
- ✅ Yaklaştırma / Uzaklaştırma  
- ✅ Renk Uzayı Dönüşümü (RGB ↔ YUV, HSV vs.)  
- ✅ Histogram Görüntüleme ve Histogram Germe  
- ✅ Aritmetik İşlemler (Görüntü Çıkarma, Çarpma)  
- ✅ Kontrast Azaltma  
- ✅ Median Filtresi ile Konvolüsyon  
- ✅ Çift Eşikleme  
- ✅ Canny Kenar Bulma (manuel adımlarla)  
- ✅ Salt & Pepper Gürültü Ekleme  
- ✅ Gürültü Temizleme (Mean, Median Filtreleri)  
- ✅ Motion Filtreleme  
- ✅ Morfolojik İşlemler (Genişleme, Aşındırma, Açma, Kapama)  

Tüm işlemler el yazımı algoritmalarla gerçekleştirilmiştir.

---

## 🧪 Kurulum

### 🔗 Gereksinimler

```bash
pip install -r requirements.txt
```

---

## 🖼️ Arayüz Görseli

![UI Görseli](https://github.com/muhammeteminayhan/image-processing-gui/blob/main/ui.png?raw=true)

---

## 📂 Kullanım

Uygulamayı başlatmak için:

```bash
python main.py
```

---

## 🧠 Notlar

- Görüntü işleme algoritmalarında hiçbir hazır OpenCV, skimage gibi kütüphane fonksiyonu kullanılmamıştır.
- Tüm işlemler temel `numpy` işlemleri ile elle kodlanmıştır.
- Sadece görüntü okuma ve gösterme işlemlerinde temel yardımcı fonksiyonlar tercih edilmiştir.

---

