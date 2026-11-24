# hemositometre-analizoru
Thoma ve Neubauer lamı için OpenCV ve Streamlit tabanlı otomatik hücre sayım ve konsantrasyon analizörü.
# 🔬 Otomatik Hemositometre Hücre Sayım & Konsantrasyon Analizörü

Thoma ve Neubauer hemositometre lamı fotoğraflarından otomatik ızgara tespiti, açı düzeltmesi ve L-kuralına göre kare içi hücre sayımı yapan Python/Streamlit uygulaması.

## Özellikler
- **Otomatik Izgara & Açı Düzeltme:** Hough dönüşümü ve morfolojik filtrelerle eğik çekilmiş lam fotoğraflarını otomatik dikleştirir ve merkez sayım karesine kilitlenir.
- **L-Kuralı (Sınır Eleme):** Üst ve sol sınıra değen hücreleri dahil eder, alt ve sağ sınıra değenleri çift sayımı önlemek için otomatik eler.
- **Esnek Laboratuvar Hesabı:** Seyreltme faktörü (1×, 1:1 Trypan Blue, 1:10 veya özel µL karışımı) ve tüp hacmine göre `Hücre/mL`, `Toplam Hücre` ve `10 µL` stok değerlerini anlık hesaplar.

## Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt
streamlit run hemositometre.py
```
