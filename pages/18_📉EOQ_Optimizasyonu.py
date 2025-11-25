# pages/18_📉Satin_Alma_Optimizasyonu.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
st.set_page_config(page_title="Satınalma Optimizasyonu", layout="wide")
make_sidebar()
yetki_kontrol("Satınalma Optimizasyonu")

# Akıllı Rehber Entegrasyonu
try:
    from help_content import yardim_goster
    yardim_goster("Satınalma Optimizasyonu")
except:
    pass

st.title("📉 Satınalma Optimizasyonu (EOQ Modeli)")
st.markdown("Stok tutma maliyetleri ile sipariş verme maliyetlerini dengeleyerek **en ekonomik sipariş miktarını** hesaplayın.")

# --- VERİ YÜKLEME ---
@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

# --- ÜRÜN SEÇİMİ VE PARAMETRELER ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Ürün Seçimi")
    # En çok satılan ürünleri listele
    top_urunler = df.groupby('UrunKodu')['Miktar'].sum().nlargest(200).index
    secilen_urun = st.selectbox("Optimize edilecek ürünü seçin:", top_urunler)
    
    # Yıllık talebi veriden otomatik hesapla (Basit projeksiyon)
    urun_df = df[df['UrunKodu'] == secilen_urun]
    toplam_satis = urun_df['Miktar'].sum()
    gun_sayisi = (urun_df['Tarih'].max() - urun_df['Tarih'].min()).days
    yillik_talep_tahmini = (toplam_satis / gun_sayisi) * 365 if gun_sayisi > 0 else 0
    
    st.info(f"📅 Veriye Göre Yıllık Tahmini Talep: **{int(yillik_talep_tahmini)}** Adet")

with col2:
    st.subheader("2. Maliyet Parametreleri")
    c1, c2, c3 = st.columns(3)
    
    D = c1.number_input("Yıllık Talep Miktarı (Adet)", value=int(yillik_talep_tahmini), min_value=1)
    S = c2.number_input("Sipariş Başına Sabit Maliyet (€)", value=50.0, help="Nakliye, gümrük, evrak işleri vb. her siparişte ödenen sabit para.")
    H = c3.number_input("Yıllık Stok Tutma Maliyeti (€/Adet)", value=2.0, help="Bir ürünü 1 yıl depoda tutmanın maliyeti (Kira, sigorta, finansman maliyeti).")

# --- EOQ HESAPLAMA ---
if H > 0 and D > 0:
    # Formül: EOQ = Kök(2 * D * S / H)
    EOQ = np.sqrt((2 * D * S) / H)
    
    siparis_sayisi = D / EOQ
    toplam_stok_maliyeti = (EOQ / 2) * H
    toplam_siparis_maliyeti = (D / EOQ) * S
    toplam_yillik_maliyet = toplam_stok_maliyeti + toplam_siparis_maliyeti
    
    st.markdown("---")
    st.subheader("📊 Optimizasyon Sonuçları")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Ekonomik Sipariş Miktarı (EOQ)", f"{int(EOQ)} Adet", help="Her seferinde sipariş etmeniz gereken en ideal miktar.")
    kpi2.metric("Yıllık Sipariş Sıklığı", f"{siparis_sayisi:.1f} Kez", help="Yılda kaç kez sipariş geçmelisiniz?")
    kpi3.metric("Minimize Edilmiş Yıllık Maliyet", f"{toplam_yillik_maliyet:,.2f} €")
    
    # --- GRAFİKSEL GÖSTERİM ---
    # Maliyet Eğrilerini Çizelim
    miktar_araligi = np.linspace(EOQ * 0.5, EOQ * 2, 100)
    
    holding_cost = (miktar_araligi / 2) * H
    ordering_cost = (D / miktar_araligi) * S
    total_cost = holding_cost + ordering_cost
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=miktar_araligi, y=holding_cost, name='Stok Tutma Maliyeti', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=miktar_araligi, y=ordering_cost, name='Sipariş Verme Maliyeti', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=miktar_araligi, y=total_cost, name='Toplam Maliyet', line=dict(color='blue', width=3)))
    
    # Optimum Noktayı İşaretle
    fig.add_trace(go.Scatter(
        x=[EOQ], y=[toplam_yillik_maliyet],
        mode='markers+text',
        name='Optimum Nokta (EOQ)',
        text=[f"EOQ: {int(EOQ)}"],
        textposition="top center",
        marker=dict(size=12, color='orange', symbol='star')
    ))

    fig.update_layout(
        title="Maliyet Optimizasyon Eğrisi",
        xaxis_title="Sipariş Miktarı (Adet)",
        yaxis_title="Yıllık Maliyet (€)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"💡 **Tavsiye:** Maliyetleri minimumda tutmak için tedarikçinizden her seferinde yaklaşık **{int(EOQ)}** adet sipariş vermeli ve bunu yılda **{int(siparis_sayisi)}** kez tekrarlamalısınız.")
else:
    st.warning("Lütfen maliyet parametrelerini (H > 0) giriniz.")