# pages/21_💶Gelecek_Nakit_Akisi.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
# Akıllı Rehber
try:
    from help_content import yardim_goster
    yardim_goster("Gelecek Nakit Akışı")
except:
    pass

st.set_page_config(page_title="Satınalma Bütçe Tahmini", layout="wide")
make_sidebar()
yetki_kontrol("Satınalma Bütçe Tahmini")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("💶 Gelecek Satınalma Bütçesi Tahmini")
st.markdown("Gelecek dönemdeki satış tahminlerine dayanarak, **ne kadarlık bir satınalma bütçesine (Nakit Çıkışı)** ihtiyacınız olacağını öngörür.")

# --- VERİ HAZIRLIĞI ---
# Aylık Maliyet Verisini Hazırla
if 'Maliyet' not in df.columns:
    df['Maliyet'] = df['BirimFiyat'] * 0.75 # Varsayılan %75 maliyet
    df['ToplamMaliyet'] = df['ToplamTutar'] * 0.75
else:
    df['ToplamMaliyet'] = df['Miktar'] * df['Maliyet']

aylik_maliyet = df.set_index('Tarih').resample('M')['ToplamMaliyet'].sum().reset_index()
aylik_maliyet.columns = ['ds', 'y'] # Prophet formatı

# --- TAHMİNLEME ---
tahmin_periyodu = st.slider("Kaç aylık bütçe tahmini yapılsın?", 3, 12, 6)

with st.spinner("Nakit akış projeksiyonu hesaplanıyor..."):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(aylik_maliyet)
    future = model.make_future_dataframe(periods=tahmin_periyodu, freq='M')
    forecast = model.predict(future)
    
    # Negatif tahminleri sıfırla
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))

# --- SONUÇLAR ---
gelecek_tahminleri = forecast.tail(tahmin_periyodu)
toplam_butce_ihtiyaci = gelecek_tahminleri['yhat'].sum()
ortalama_aylik_butce = gelecek_tahminleri['yhat'].mean()

st.markdown("---")
col1, col2 = st.columns(2)
col1.metric(f"Gelecek {tahmin_periyodu} Ay İçin Toplam Bütçe İhtiyacı", f"{toplam_butce_ihtiyaci:,.0f} €", help="Tahmini satışları karşılamak için yapılması gereken stok alımı.")
col2.metric("Ortalama Aylık Nakit Çıkışı", f"{ortalama_aylik_butce:,.0f} €")

# --- GRAFİK ---
st.subheader("Tahmini Aylık Nakit Çıkış Grafiği")

fig = go.Figure()

# Geçmiş Veri
gecmis_veri = forecast[:-tahmin_periyodu]
fig.add_trace(go.Scatter(x=gecmis_veri['ds'], y=gecmis_veri['yhat'], name='Gerçekleşen Maliyetler', line=dict(color='gray')))

# Gelecek Tahmin
fig.add_trace(go.Scatter(x=gelecek_tahminleri['ds'], y=gelecek_tahminleri['yhat'], name='Tahmini Bütçe İhtiyacı', 
                         line=dict(color='red', width=3, dash='dot')))

# Güven Aralığı
fig.add_trace(go.Scatter(x=gelecek_tahminleri['ds'], y=gelecek_tahminleri['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=gelecek_tahminleri['ds'], y=gelecek_tahminleri['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', name='Güven Aralığı'))

fig.update_layout(title="Satınalma Nakit Akış Projeksiyonu", xaxis_title="Tarih", yaxis_title="Tutar (€)")
st.plotly_chart(fig, use_container_width=True)

# --- TABLO ---
with st.expander("Aylık Detaylı Bütçe Tablosunu Görüntüle"):
    gosterim_df = gelecek_tahminleri[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    gosterim_df.columns = ['Ay', 'Tahmini Bütçe', 'Min. İhtiyaç', 'Max. Risk']
    gosterim_df['Ay'] = gosterim_df['Ay'].dt.strftime('%B %Y')
    st.dataframe(gosterim_df.style.format({'Tahmini Bütçe': '{:,.0f} €', 'Min. İhtiyaç': '{:,.0f} €', 'Max. Risk': '{:,.0f} €'}))