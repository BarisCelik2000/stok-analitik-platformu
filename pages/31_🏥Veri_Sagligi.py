# pages/31_🏥Veri_Sagligi.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar
st.set_page_config(page_title="Veri Sağlığı Panosu", layout="wide")
make_sidebar()
yetki_kontrol("Veri Sağlığı Panosu")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🏥 Veri Sağlığı Panosu (Data Health Check)")
st.markdown("""
Bu modül, analizlerde kullanılan verinin kalitesini, tutarlılığını ve güvenilirliğini ölçer.
Yüksek bir skor, analiz sonuçlarına güvenebileceğiniz anlamına gelir.
""")

# --- SAĞLIK KONTROL MOTORU ---
toplam_satir = len(df)
toplam_hucre = df.size

# 1. Eksik Veri Kontrolü
eksik_degerler = df.isnull().sum().sum()
eksiklik_orani = (eksik_degerler / toplam_hucre) * 100

# 2. Mükerrer (Tekrar Eden) Kayıt Kontrolü
tekrar_eden = df.duplicated().sum()
tekrar_orani = (tekrar_eden / toplam_satir) * 100

# 3. Mantıksal Hata Kontrolü (Negatif Fiyat vb.)
negatif_fiyat = len(df[df['BirimFiyat'] < 0])
sifir_fiyat = len(df[df['BirimFiyat'] == 0])
mantiksal_hata_orani = ((negatif_fiyat + sifir_fiyat) / toplam_satir) * 100

# --- SKOR HESAPLAMA ---
# Başlangıç: 100 Puan
saglik_skoru = 100
saglik_skoru -= (eksiklik_orani * 5)  # Her %1 eksiklik için 5 puan kır
saglik_skoru -= (tekrar_orani * 10)   # Her %1 tekrar için 10 puan kır
saglik_skoru -= (mantiksal_hata_orani * 10) # Her %1 mantık hatası için 10 puan kır

# Skor 0'ın altına düşmesin, 100'ü geçmesin
saglik_skoru = max(0, min(100, saglik_skoru))

# Skor Rengi
renk = "green" if saglik_skoru >= 90 else "orange" if saglik_skoru >= 70 else "red"

# --- GÖSTERGE PANELİ (GAUGE) ---
col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = saglik_skoru,
        title = {'text': "Genel Veri Sağlık Skoru"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': renk},
            'steps': [
                {'range': [0, 70], 'color': "lightgray"},
                {'range': [70, 90], 'color': "whitesmoke"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.subheader("Teşhis Özeti")
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Satır Sayısı", f"{toplam_satir:,}")
    c2.metric("Eksik Hücre Sayısı", f"{eksik_degerler}", f"%{eksiklik_orani:.2f} Oran", delta_color="inverse")
    c3.metric("Tekrar Eden Satır", f"{tekrar_eden}", f"%{tekrar_orani:.2f} Oran", delta_color="inverse")
    
    st.markdown("---")
    if saglik_skoru == 100:
        st.success("✅ Mükemmel! Veri setiniz tertemiz, analize tamamen uygun.")
    elif saglik_skoru >= 90:
        st.info("👍 İyi Durumda. Ufak tefek eksikler var ama analizi etkilemez.")
    else:
        st.warning("⚠️ Dikkat! Veri kalitesi düşük. Aşağıdaki detayları inceleyip veri kaynağını düzeltmeniz önerilir.")

st.markdown("---")

# --- DETAYLI ANALİZ SEKMELERİ ---
tab1, tab2, tab3 = st.tabs(["Sütun Kalitesi", "Aykırı Değerler (Outliers)", "Veri Tipleri"])

with tab1:
    st.subheader("Sütun Bazlı Kalite Kontrolü")
    
    # Her sütun için boşluk ve benzersizlik analizi
    kalite_df = pd.DataFrame({
        'Sütun Adı': df.columns,
        'Veri Tipi': df.dtypes.values.astype(str),
        'Dolu Veri (%)': [(1 - (df[col].isnull().sum() / len(df))) * 100 for col in df.columns],
        'Benzersiz Değer Sayısı': [df[col].nunique() for col in df.columns],
        'Örnek Değer': [df[col].iloc[0] if len(df) > 0 else "-" for col in df.columns]
    })
    
    st.dataframe(
        kalite_df.style.format({'Dolu Veri (%)': '{:.1f}%'})
                 .background_gradient(cmap='RdYlGn', subset=['Dolu Veri (%)'])
    )

with tab2:
    st.subheader("Aykırı Değer (Outlier) Tespiti")
    st.markdown("İstatistiksel olarak normalin çok dışında kalan (Z-Score > 3) şüpheli işlemler.")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown("**Birim Fiyat Aykırılıkları**")
        fig_box_fiyat = px.box(df, y="BirimFiyat", title="Fiyat Dağılımı (Box Plot)")
        st.plotly_chart(fig_box_fiyat, use_container_width=True)
        
    with col_out2:
        st.markdown("**Miktar (Adet) Aykırılıkları**")
        fig_box_miktar = px.box(df, y="Miktar", title="Satış Adedi Dağılımı")
        st.plotly_chart(fig_box_miktar, use_container_width=True)

with tab3:
    st.subheader("Veri Tipleri ve Bellek Kullanımı")
    buffer = df.info(buf=None) # Streamlit'e info basmak zordur, manuel yapalım
    
    mem_usage = df.memory_usage(deep=True).sum() / 1024**2 # MB cinsinden
    
    col_mem1, col_mem2 = st.columns(2)
    col_mem1.metric("Toplam Bellek Kullanımı", f"{mem_usage:.2f} MB")
    col_mem2.metric("Sütun Sayısı", len(df.columns))
    
    st.caption("Veri tiplerinin doğruluğu (Tarih sütununun 'datetime', Fiyatın 'float' olması vb.) analiz motorunun çalışması için kritiktir.")