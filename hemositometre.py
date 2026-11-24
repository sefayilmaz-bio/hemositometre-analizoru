import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Otomatik Hemositometre Analizörü", layout="wide")
st.title("🔬 Otomatik Hemositometre Hücre Sayım & Konsantrasyon Analizörü")
st.write("Thoma ve Neubauer lamı fotoğraflarında ızgarayı otomatik hizalar, L-kuralına göre kare içi hücre sayımı ve laboratuvar konsantrasyon hesabını yapar.")

dosya = st.file_uploader(
    "Hemositometre (Thoma / Neubauer) fotoğrafını yükleyin (.jpg, .png):",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.header("🧪 Deney & Hesaplama Protokolü")
kare_tipi = st.sidebar.selectbox(
    "Sayılan Kare Tipi:",
    [
        "1 Büyük Kare (1×1 mm = 0.1 µL → ×10⁴)",
        "1 Küçük İç Kare (0.2×0.2 mm = 0.004 µL → ×2.5×10⁵)"
    ],
    help="Thoma merkez karesi veya Neubauer'in 4 köşe karesinden biri için '1 Büyük Kare' seçin."
)

seyreltme_modu = st.sidebar.selectbox(
    "Seyreltme / Boya Protokolü:",
    [
        "Seyreltme Yok (Direkt Örnek → 1×)",
        "Trypan Blue 1:1 (10 µL hücre + 10 µL boya → 2×)",
        "1:10 Seyreltme (10 µL hücre + 90 µL PBS → 10×)",
        "Özel Karışım (µL girerek hesapla)"
    ]
)

if "Özel" in seyreltme_modu:
    c_u1, c_u2 = st.sidebar.columns(2)
    hucre_ul = c_u1.number_input("Hücre (µL):", min_value=1.0, value=10.0, step=5.0)
    boya_ul = c_u2.number_input("Boya/PBS (µL):", min_value=0.0, value=10.0, step=5.0)
    seyreltme = (hucre_ul + boya_ul) / hucre_ul
    st.sidebar.caption(f"Hesaplanan Seyreltme Faktörü (DF): **{seyreltme:.2f}×**")
elif "1:1" in seyreltme_modu:
    seyreltme = 2.0
elif "1:10" in seyreltme_modu:
    seyreltme = 10.0
else:
    seyreltme = 1.0

c_h1, c_h2 = st.sidebar.columns([2, 1])
hacim_deger = c_h1.number_input("Ana Tüp Hacmi:", min_value=0.1, value=1.0, step=0.5)
hacim_birim = c_h2.selectbox("Birim:", ["mL", "µL"])
toplam_ml = hacim_deger if hacim_birim == "mL" else hacim_deger / 1000.0

st.sidebar.header("📐 Sayım Karesi & Izgara")
oto_izgara = st.sidebar.checkbox("🎯 Izgarayı ve kareyi OTOMATİK bul", value=True)
l_kurali = st.sidebar.checkbox("L-Kuralı (Üst/sol dahil, alt/sağ hariç)", value=True)

if oto_izgara:
    ek_aci = st.sidebar.slider("İnce Açı Ayarı (°):", -10.0, 10.0, 0.0, 0.5)
    kaydir_x = st.sidebar.slider("İnce Kaydır Yatay (px):", -150, 150, 0, 2)
    kaydir_y = st.sidebar.slider("İnce Kaydır Dikey (px):", -150, 150, 0, 2)
    kenar_payi = st.sidebar.slider("Kare Genişlet / Daralt (px):", -80, 80, 0, 2)
else:
    manuel_aci = st.sidebar.slider("Izgara Açısı (°):", -45.0, 45.0, 0.0, 0.5)
    kare_x = st.sidebar.slider("Kare Merkez X (%):", 10, 90, 50)
    kare_y = st.sidebar.slider("Kare Merkez Y (%):", 10, 90, 50)
    kare_boyut = st.sidebar.slider("Kare Boyutu (%):", 10, 85, 36)

st.sidebar.header("🎛️ Hücre Algılama Ayarları")
hucre_kontrast = st.sidebar.radio(
    "Hücre Görünümü:",
    ["Parlak Hücreler (Klasik / Faz Kontrast)", "Koyu Hücreler (Boyalı / Trypan Blue)"]
)
izgara_silici = st.sidebar.checkbox("Izgara çizgilerini arka plandan sil", value=True)
numara_goster = st.sidebar.checkbox("Hücre numaralarını göster", value=True)
hassasiyet = st.sidebar.slider("Hassasiyet (Düşük = Daha çok hücre):", 5, 100, 18)
hucre_capi = st.sidebar.slider("Tahmini Hücre Çapı (px):", 3, 25, 7)
min_alan = st.sidebar.slider("Min Hücre Alanı (px²):", 2, 300, 5)
max_alan = st.sidebar.slider("Max Hücre Alanı (px²):", 30, 5000, 260)
min_dairesellik = st.sidebar.slider("Dairesellik Filtresi:", 0.0, 0.90, 0.30, 0.05)

def dondur(img, aci):
    if abs(aci) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), aci, 1.0)
    border = (0, 0, 0) if img.ndim == 3 else 0
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

def okuler_maskesi(gray):
    h, w = gray.shape
    koseler = [gray[0:15, 0:15].mean(), gray[0:15, -15:].mean(), gray[-15:, 0:15].mean(), gray[-15:, -15:].mean()]
    if np.mean(koseler) < 35:
        _, m = cv2.threshold(gray, 28, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            temiz = np.zeros_like(gray)
            cv2.drawContours(temiz, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
            pay = max(15, int(min(h, w) * 0.06)) | 1
            temiz = cv2.erode(temiz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pay, pay)))
            if cv2.countNonZero(temiz) > 500:
                return temiz
    return np.full_like(gray, 255)

def en_guclu_aralik(profil, max_bosluk):
    pmax = float(profil.max())
    idx = np.where(profil > pmax * 0.18)[0] if pmax > 1e-6 else []
    if len(idx) < 5:
        return None
    bloklar, bas, onceki = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - onceki > max_bosluk:
            bloklar.append((bas, onceki))
            bas = i
        onceki = i
    bloklar.append((bas, onceki))
    en_iyi = max(bloklar, key=lambda b: float(profil[b[0]:b[1] + 1].sum()))
    return (int(en_iyi[0]), int(en_iyi[1])) if en_iyi[1] - en_iyi[0] >= 20 else None

def otomatik_izgara_bul(gray):
    h, w = gray.shape
    mask = okuler_maskesi(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    hat = np.maximum(cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, k9), cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, k9))
    hat = cv2.bitwise_and(hat, hat, mask=mask)
    p88 = np.percentile(hat[mask > 0], 88) if np.any(mask > 0) else 20
    c1 = cv2.Canny(hat, max(8, p88 * 0.45), max(20, p88 * 1.4))
    edges = cv2.bitwise_and(c1, c1, mask=mask)
    min_len = max(40, int(min(h, w) * 0.18))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=45, minLineLength=min_len, maxLineGap=20)
    if lines is None or lines.size < 8:
        c2 = cv2.Canny(blur, 25, 80)
        lines = cv2.HoughLinesP(cv2.bitwise_and(c2, c2, mask=mask), 1, np.pi / 360, threshold=45, minLineLength=min_len, maxLineGap=20)
    aci = 0.0
    if lines is not None and lines.size >= 4:
        cizgiler = lines.reshape(-1, 4).astype(np.float64)
        acilar = [(np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 45.0) % 90.0 - 45.0 for x1, y1, x2, y2 in cizgiler]
        agirlik = [np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in cizgiler]
        s = np.argsort(acilar)
        cs = np.cumsum(np.array(agirlik)[s])
        med_i = min(int(np.searchsorted(cs, cs[-1] * 0.5)), len(acilar) - 1)
        aci = float(np.round(np.array(acilar)[s][med_i], 1))

    g_duz = dondur(gray, aci)
    m_f = (okuler_maskesi(g_duz) > 0).astype(np.float32)
    gx = cv2.dilate(np.abs(cv2.Sobel(g_duz, cv2.CV_32F, 1, 0, ksize=3)) * m_f, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    gy = cv2.dilate(np.abs(cv2.Sobel(g_duz, cv2.CV_32F, 0, 1, ksize=3)) * m_f, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    L = max(35, int(min(h, w) * 0.11))
    col_p = cv2.morphologyEx(gx, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))).sum(axis=0)
    row_p = cv2.morphologyEx(gy, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))).sum(axis=1)
    mb = max(25, int(min(h, w) * 0.08))
    xa, ya = en_guclu_aralik(col_p, mb), en_guclu_aralik(row_p, mb)
    return (aci, (xa[0], ya[0], xa[1], ya[1])) if (xa and ya) else (aci, None)

if not dosya:
    st.info("⬆️ Başlamak için yukarıdaki kutuya bir hemositometre (Thoma veya Neubauer) fotoğrafı yükleyin.")
else:
    rgb_img = np.array(Image.open(dosya).convert("RGB"))
    h0, w0 = rgb_img.shape[:2]
    if max(h0, w0) > 1400:
        oran = 1400 / max(h0, w0)
        rgb_img = cv2.resize(rgb_img, (int(w0 * oran), int(h0 * oran)))

    gray_ham = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    h, w = gray_ham.shape

    if oto_izgara:
        bulunan_aci, kutu = otomatik_izgara_bul(gray_ham)
        rgb_img = dondur(rgb_img, bulunan_aci + ek_aci)
        gray_full = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        if kutu:
            bx1, by1, bx2, by2 = kutu
            x1, y1 = max(0, bx1 + kaydir_x - kenar_payi), max(0, by1 + kaydir_y - kenar_payi)
            x2, y2 = min(w, bx2 + kaydir_x + kenar_payi), min(h, by2 + kaydir_y + kenar_payi)
            st.success(f"🎯 Izgara Otomatik Kilitlendi! Algılanan Açı: **{bulunan_aci:+.1f}°** | Kare Boyutu: **{x2-x1}×{y2-y1} px**")
        else:
            st.warning("⚠️ Bu fotoğrafta hemositometre ızgarası otomatik bulunamadı. Sol menüden 'OTOMATİK bul' tikini kaldırıp kareyi elle yerleştirebilirsin.")
            x1, y1, x2, y2 = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
    else:
        rgb_img = dondur(rgb_img, manuel_aci)
        gray_full = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        yari = int(min(h, w) * kare_boyut / 200)
        cx_b, cy_b = int(w * kare_x / 100), int(h * kare_y / 100)
        x1, x2, y1, y2 = max(0, cx_b - yari), min(w, cx_b + yari), max(0, cy_b - yari), min(h, cy_b + yari)

    if x2 - x1 < 20 or y2 - y1 < 20:
        st.error("Seçili kare çok dar, sol menüden genişlet.")
        st.stop()

    mask_full = okuler_maskesi(gray_full)
    roi_rgb, gray, mask = rgb_img[y1:y2, x1:x2], gray_full[y1:y2, x1:x2], mask_full[y1:y2, x1:x2]
    roi_h, roi_w = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hucre_capi * 2 + 1, hucre_capi * 2 + 1))

    if "Parlak" in hucre_kontrast:
        if izgara_silici:
            cu = max(17, hucre_capi * 3 + 1)
            bg = np.maximum(
                cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (cu, 1))),
                cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, cu)))
            )
            vurgulu = cv2.subtract(gray, bg)
        else:
            vurgulu = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    else:
        if izgara_silici:
            cu = max(17, hucre_capi * 3 + 1)
            bg = np.minimum(
                cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (cu, 1))),
                cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, cu)))
            )
            vurgulu = cv2.subtract(bg, gray)
        else:
            vurgulu = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    vurgulu = cv2.bitwise_and(vurgulu, vurgulu, mask=mask)
    _, thresh = cv2.threshold(vurgulu, hassasiyet, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    konturlar, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gecerli, haric, alanlar = [], [], []
    for c in konturlar:
        alan, cevre = cv2.contourArea(c), cv2.arcLength(c, True)
        if not (min_alan <= alan <= max_alan) or cevre == 0 or (4 * np.pi * alan / (cevre * cevre)) < min_dairesellik:
            continue
        xb, yb, wb, hb = cv2.boundingRect(c)
        if l_kurali and (xb + wb >= roi_w - 2 or yb + hb >= roi_h - 2):
            haric.append(c)
        else:
            gecerli.append(c)
            alanlar.append(alan)

    isaretli = roi_rgb.copy()
    for idx, c in enumerate(gecerli, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cv2.circle(isaretli, (int(cx), int(cy)), max(4, int(r) + 2), (0, 255, 80), 2)
        if numara_goster:
            cv2.putText(isaretli, str(idx), (int(cx) - 5, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for c in haric:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cv2.circle(isaretli, (int(cx), int(cy)), max(4, int(r) + 2), (255, 60, 60), 2)

    genel = rgb_img.copy()
    cv2.line(genel, (x1, y1), (x2, y1), (0, 255, 80), 3)
    cv2.line(genel, (x1, y1), (x1, y2), (0, 255, 80), 3)
    cv2.line(genel, (x1, y2), (x2, y2), (255, 60, 60), 3)
    cv2.line(genel, (x2, y1), (x2, y2), (255, 60, 60), 3)

    carpan = 10_000 if "1 Büyük Kare" in kare_tipi else 250_000
    kare_hacim_ul = 0.1 if "1 Büyük Kare" in kare_tipi else 0.004
    hucre_per_ml = len(gecerli) * seyreltme * carpan
    toplam_hucre = hucre_per_ml * toplam_ml
    on_ul_hucre = hucre_per_ml * 0.01  # 10 µL = 0.01 mL

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kare İçi Hücre", len(gecerli), help=f"L-kuralıyla elenen alt/sağ sınır hücresi: {len(haric)}")
    m2.metric("Konsantrasyon (Hücre/mL)", f"{hucre_per_ml:,.0f}".replace(",", "."))
    m3.metric(f"Toplam ({hacim_deger:g} {hacim_birim})", f"{toplam_hucre:,.0f}".replace(",", "."))
    m4.metric("10 µL Stoktaki Hücre", f"{on_ul_hucre:,.0f}".replace(",", "."))

    st.caption(
        (
            f"📐 **Canlı Formül:** `{len(gecerli)} hücre` × `{seyreltme:g}× (Seyreltme)` × `{carpan:,}` "
            f"(Sayılan kare: {kare_hacim_ul:g} µL → 1 mL) = **{hucre_per_ml:,.0f} hücre/mL**"
        ).replace(",", ".")
    )

    c1, c2 = st.columns(2)
    c1.subheader("1. Sayım Karesi Hizalama")
    c1.image(genel, caption="Yeşil kenar (üst/sol): DAHİL | Kırmızı kenar (alt/sağ): HARİÇ")
    c2.subheader("2. Kare İçi Hücre Sayımı (Yakın Çekim)")
    c2.image(isaretli, caption=f"Yeşil: {len(gecerli)} sayılan | Kırmızı: {len(haric)} L-kuralıyla elenen")

    with st.expander("🔍 Siyah-Beyaz Maske (İnce Ayar Kontrolü)"):
        st.image(thresh, caption="Izgara çizgileri görünüyorsa Hassasiyeti yükselt veya İnce Açı Ayarını ±0.5° oyna.")
