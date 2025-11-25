# pages/20_🛡️Guvenlik_Stogu_ve_ROP.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
# Akıllı Rehber
try:
    from help_content import yardim_goster
    # help_content.py'ye bu başlığı eklemeniz gerekebilir, şimdilik pass geçelim
    yardim_goster("Güvenlik Stoğu ve ROP") 
except:
    pass

st.set_page_config(page_title="Sipariş Tetikleme ve Güvenlik Stoğu", layout="wide")
make_sidebar()
yetki_kontrol("Sipariş Tetikleme ve Güvenlik Stoğu")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🛡️ Sipariş Tetikleme Noktası (ROP) ve Güvenlik Stoğu")
st.markdown("""
Bu modül, stoksuz kalmamanız için sipariş vermeniz gereken kritik stok seviyesini (**Reorder Point**) hesaplar.
* **ROP:** Stok seviyesi bu sayıya düştüğünde sipariş vermelisiniz.
* **Güvenlik Stoğu:** Beklenmedik talep artışlarına veya tedarik gecikmelerine karşı tampon.
""")

# --- ÜRÜN SEÇİMİ VE PARAMETRELER ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Ürün Analizi")
    # En çok satılan ürünleri listele
    top_urunler = df.groupby('UrunKodu')['Miktar'].sum().nlargest(200).index
    secilen_urun = st.selectbox("Analiz edilecek ürünü seçin:", top_urunler)
    
    # Veriden Otomatik Hesaplamalar
    urun_df = df[df['UrunKodu'] == secilen_urun].copy()
    
    # Günlük Satış İstatistikleri
    # Veriyi günlük bazda gruplayalım (Satış olmayan günleri de 0 olarak eklemek gerekir ama basitleştirilmiş versiyon)
    gunluk_satis = urun_df.groupby('Tarih')['Miktar'].sum()
    
    ortalama_gunluk_satis = gunluk_satis.mean()
    maksimum_gunluk_satis = gunluk_satis.max()
    std_dev_satis = gunluk_satis.std()
    
    st.info(f"""
    📊 **Veri İstatistikleri:**
    * Ort. Günlük Satış: **{ortalama_gunluk_satis:.1f}** Adet
    * Max. Günlük Satış: **{maksimum_gunluk_satis:.1f}** Adet
    * Standart Sapma: **{std_dev_satis:.1f}**
    """)

with col2:
    st.subheader("2. Tedarik Süreleri (Lead Time)")
    
    c1, c2 = st.columns(2)
    lead_time_avg = c1.number_input("Ortalama Teslim Süresi (Gün)", value=14, help="Siparişi verdikten kaç gün sonra mal depoya giriyor?")
    lead_time_max = c2.number_input("Maksimum Teslim Süresi (Gün)", value=21, help="Tedarikçinin en kötü durumdaki gecikmeli teslim süresi.")
    
    st.markdown("---")
    st.subheader("3. Servis Seviyesi Hedefi")
    service_level = st.slider("Hedeflenen Servis Seviyesi (%)", 80, 99, 95, 
                              help="Müşteri talebinin % kaçını stoktan anında karşılamak istiyorsunuz? Yüksek oran = Yüksek stok maliyeti.")
    
    # Z-Skoru (Normal Dağılım Tablosundan)
    # %90 -> 1.28, %95 -> 1.645, %99 -> 2.33
    z_score_map = {80: 0.84, 85: 1.04, 90: 1.28, 95: 1.645, 98: 2.05, 99: 2.33}
    # Yaklaşık değer için en yakın key'i bulalım (Slider aralığına göre)
    z_val = z_score_map.get(service_level, 1.645) 

# --- HESAPLAMALAR ---
# 1. Güvenlik Stoğu (Safety Stock)
# Formül: (Max Günlük Satış * Max Lead Time) - (Ort Günlük Satış * Ort Lead Time)
# Veya daha istatistiksel yöntem: Z * StdDev * Sqrt(Lead Time)
# Biz daha güvenli olan "Max - Ort" yöntemini (Konservatif) veya Z-skorlu yöntemi kullanabiliriz.
# Z-Skorlu yöntem daha profesyoneldir:
guvenlik_stogu = z_val * std_dev_satis * np.sqrt(lead_time_avg)

# 2. Reorder Point (ROP)
# ROP = (Ortalama Günlük Satış * Ortalama Lead Time) + Güvenlik Stoğu
talep_lead_time_boyunca = ortalama_gunluk_satis * lead_time_avg
rop = talep_lead_time_boyunca + guvenlik_stogu

st.markdown("---")
st.subheader("🚨 Hesaplama Sonuçları")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Sipariş Tetikleme Noktası (ROP)", f"{int(rop)} Adet", help="Stok bu seviyeye düştüğü AN sipariş geçmelisiniz.")
kpi2.metric("Güvenlik Stoğu", f"{int(guvenlik_stogu)} Adet", help="Tedarikçi gecikirse veya talep patlarsa sizi koruyacak tampon stok.")
kpi3.metric("Lead Time Talebi", f"{int(talep_lead_time_boyunca)} Adet", help="Ürün yoldayken satacağınız tahmini miktar.")

# --- GÖRSELLEŞTİRME (STOK SİMÜLASYONU) ---
st.subheader("📉 Stok Tüketim Simülasyonu")

# Basit bir testere dişi grafiği (Sawtooth inventory model) simüle edelim
gunler = list(range(0, 60))
stok_seviyesi = []
siparis_miktari = rop * 1.5 # Örnek sipariş miktarı (EOQ'dan gelebilirdi)
mevcut_stok = siparis_miktari + guvenlik_stogu # Başlangıç

siparis_verildi = False
siparis_bekleme_gunu = 0

for gun in gunler:
    # Günlük satış kadar düş (Ortalama)
    # Biraz rastgelelik ekleyelim
    gunluk_satis_sim = np.random.normal(ortalama_gunluk_satis, std_dev_satis)
    gunluk_satis_sim = max(0, gunluk_satis_sim)
    
    mevcut_stok -= gunluk_satis_sim
    
    # Sipariş yönetimi
    if siparis_verildi:
        siparis_bekleme_gunu += 1
        if siparis_bekleme_gunu >= lead_time_avg:
            mevcut_stok += siparis_miktari
            siparis_verildi = False
            siparis_bekleme_gunu = 0
    
    elif mevcut_stok <= rop:
        siparis_verildi = True
        siparis_bekleme_gunu = 0
        
    stok_seviyesi.append(max(0, mevcut_stok))

fig = go.Figure()
fig.add_trace(go.Scatter(x=gunler, y=stok_seviyesi, name='Stok Seviyesi', fill='tozeroy', line=dict(color='#636EFA')))
fig.add_hline(y=rop, line_dash="dash", line_color="orange", annotation_text="Sipariş Noktası (ROP)")
fig.add_hline(y=guvenlik_stogu, line_dash="dot", line_color="red", annotation_text="Güvenlik Stoğu")

fig.update_layout(
    title="Stok Döngüsü Simülasyonu (Gelecek 60 Gün)",
    xaxis_title="Günler",
    yaxis_title="Stok Adedi",
    hovermode="x"
)
st.plotly_chart(fig, use_container_width=True)

st.success(f"""
💡 **Yönetici Özeti:** Bu ürün için depoda **{int(rop)}** adet kaldığında, tedarikçinize yeni sipariş geçmelisiniz. 
Bu sipariş gelene kadar elinizdeki stok (ortalama olarak) tükenecek ve geriye sadece risklere karşı **{int(guvenlik_stogu)}** adetlik tampon stoğunuz kalacaktır.
""")