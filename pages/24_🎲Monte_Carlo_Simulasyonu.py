# pages/24_🎲Monte_Carlo_Simulasyonu.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar

st.set_page_config(page_title="Monte Carlo Simülasyonu", layout="wide")
make_sidebar()
yetki_kontrol("Monte Carlo Simülasyonu")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🎲 Monte Carlo Bütçe ve Risk Simülasyonu")
st.markdown("""
Bu modül, geçmiş satış verilerinizin oynaklığını (volatilitesini) kullanarak, 
gelecek dönem cirosunun hangi aralıklarda olacağını **binlerce senaryo** ile test eder.
""")

# --- VERİ HAZIRLIĞI ---
# Günlük satış toplamlarını bulalım
gunluk_satis = df.groupby('Tarih')['ToplamTutar'].sum()

# İstatistiksel Parametreler (Ortalama ve Standart Sapma)
mu = gunluk_satis.mean()   # Ortalama Günlük Ciro
sigma = gunluk_satis.std() # Günlük Oynaklık (Risk)

col1, col2, col3 = st.columns(3)
col1.metric("Ortalama Günlük Ciro", f"{mu:,.0f} €")
col2.metric("Günlük Oynaklık (Std Sapma)", f"{sigma:,.0f} €")
col3.info("Geçmiş veriye dayalı temel parametreler.")

st.markdown("---")

# --- SİMÜLASYON AYARLARI ---
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    simulasyon_gunu = st.slider("Kaç Günlük Tahmin Yapılsın?", 7, 90, 30)
with col_sim2:
    senaryo_sayisi = st.slider("Kaç Farklı Senaryo Üretilsin?", 100, 5000, 1000, step=100)

if st.button("Simülasyonu Başlat 🎲", type="primary"):
    with st.spinner(f"{senaryo_sayisi} farklı gelecek senaryosu hesaplanıyor..."):
        
        # Monte Carlo Motoru
        # Her senaryo için: (Gün Sayısı) kadar rastgele sayı üret (Ortalama ve Sapmaya göre)
        simulasyonlar = []
        toplam_cirolar = []
        
        np.random.seed(42) # Tekrarlanabilirlik için
        
        for i in range(senaryo_sayisi):
            # Normal dağılıma uygun rastgele günlük cirolar üret
            gunluk_tahminler = np.random.normal(mu, sigma, simulasyon_gunu)
            # Negatif satış olamayacağı için 0 ile sınırla
            gunluk_tahminler = np.maximum(0, gunluk_tahminler)
            
            # Kümülatif (Birikimli) ciro büyümesi
            kumulatif_ciro = gunluk_tahminler.cumsum()
            
            simulasyonlar.append(kumulatif_ciro)
            toplam_cirolar.append(kumulatif_ciro[-1]) # O senaryonun sonundaki toplam ciro

        # --- SONUÇ ANALİZİ ---
        toplam_cirolar = np.array(toplam_cirolar)
        
        # Olasılık Aralıkları (Percentiles)
        p5 = np.percentile(toplam_cirolar, 5)   # Kötü Senaryo (%95 ihtimalle bundan iyi olacak)
        p50 = np.percentile(toplam_cirolar, 50) # Beklenen Senaryo (Medyan)
        p95 = np.percentile(toplam_cirolar, 95) # İyi Senaryo (%5 ihtimalle buna ulaşabiliriz)
        
        st.subheader("📊 Simülasyon Sonuçları")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Kötü Senaryo (Min. Hedef)", f"{p5:,.0f} €", help="En kötü durumda bile cironun bu seviyenin altına düşme ihtimali sadece %5.")
        kpi2.metric("Beklenen Ciro (Medyan)", f"{p50:,.0f} €", help="En olası sonuç.")
        kpi3.metric("İyimser Senaryo (Fırsat)", f"{p95:,.0f} €", help="İşler çok iyi giderse ulaşılabilecek seviye.")
        
        # --- GRAFİKLER ---
        tab_g1, tab_g2 = st.tabs(["Olasılık Dağılımı (Histogram)", "Senaryo Yolları (Spagetti Grafik)"])
        
        with tab_g1:
            fig_hist = px.histogram(toplam_cirolar, nbins=30, title=f"Gelecek {simulasyon_gunu} Günlük Toplam Ciro Olasılıkları")
            fig_hist.add_vline(x=p5, line_dash="dash", line_color="red", annotation_text="Kötü Senaryo")
            fig_hist.add_vline(x=p50, line_dash="solid", line_color="green", annotation_text="Beklenen")
            fig_hist.add_vline(x=p95, line_dash="dash", line_color="blue", annotation_text="İyi Senaryo")
            fig_hist.update_layout(xaxis_title="Tahmini Toplam Ciro (€)", yaxis_title="Senaryo Sıklığı", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab_g2:
            # Performans için sadece ilk 100 senaryoyu çizelim
            fig_lines = go.Figure()
            x_axis = list(range(1, simulasyon_gunu + 1))
            
            for i in range(min(100, senaryo_sayisi)):
                fig_lines.add_trace(go.Scatter(x=x_axis, y=simulasyonlar[i], mode='lines', line=dict(width=1, color='rgba(100, 100, 100, 0.1)'), showlegend=False))
            
            # Ortalama yolu ekle
            ortalama_yol = np.mean(simulasyonlar, axis=0)
            fig_lines.add_trace(go.Scatter(x=x_axis, y=ortalama_yol, mode='lines', name='Ortalama Yol', line=dict(color='red', width=3)))
            
            fig_lines.update_layout(title="Olası Ciro Gelişim Yolları (İlk 100 Örnek)", xaxis_title="Gün", yaxis_title="Kümülatif Ciro (€)")
            st.plotly_chart(fig_lines, use_container_width=True)