# pages/0_📝Ozet_Rapor.py

import streamlit as st
import plotly.express as px
import pandas as pd
from data_handler import veriyi_yukle_ve_temizle, genel_satis_trendi_hazirla
from analysis_engine import (rfm_skorlarini_hesapla, 
                           musterileri_segmentle, 
                           churn_tahmin_modeli_olustur, 
                           clv_hesapla,
                           anomali_tespiti_yap,
                           davranissal_anomali_tespiti_yap,
                           otomatik_ozet_uret)


from navigation import make_sidebar
from auth_manager import yetki_kontrol

st.set_page_config(page_title="Yönetici Özet Raporu", layout="wide")

# Güvenlik ve Navigasyon
make_sidebar()
yetki_kontrol("Özet Rapor")

try:
    from help_content import yardim_goster
    yardim_goster("Genel Bakış") # İçeriği aynı olduğu için eski yardımı kullanabiliriz
except:
    pass
# -----------------------------------------------------

# --- VERİ YÜKLEME ---
@st.cache_data
def veriyi_getir_ve_isle():
    dosya_adi = 'satis_verileri_guncellenmis.json' 
    temiz_df = veriyi_yukle_ve_temizle(dosya_adi)
    rfm_df = rfm_skorlarini_hesapla(temiz_df)
    segmentli_df = musterileri_segmentle(rfm_df)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(segmentli_df)
    clv_df = clv_hesapla(churn_df)
    sonuclar_df = clv_df.copy()
    if 'MusteriAdi' not in sonuclar_df.columns:
        sonuclar_df['MusteriAdi'] = sonuclar_df.index
        
    # Anomali listelerini session state'e atalım (İkonlar için)
    if 'profil_anomalileri' not in st.session_state:
        profil_anomalileri_df = anomali_tespiti_yap(sonuclar_df.copy())
        st.session_state.profil_anomalileri = profil_anomalileri_df[profil_anomalileri_df['Anomali_Etiketi'] == -1].index.tolist()
    
    if 'davranissal_anomaliler' not in st.session_state:
        davranissal_anomaliler_df = davranissal_anomali_tespiti_yap(temiz_df)
        st.session_state.davranissal_anomaliler = davranissal_anomaliler_df['MusteriID'].tolist()
        
    return temiz_df, sonuclar_df

try:
    temiz_df, sonuclar_df = veriyi_getir_ve_isle()
except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

st.title("📝 Yönetici Özet Raporu")
st.markdown("Şirketin genel performans metrikleri, satış trendleri ve segment dağılımları.")

# --- FİLTRELER ---
with st.expander("⚙️ Rapor Filtreleri", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_tarih = temiz_df['Tarih'].min()
        max_tarih = temiz_df['Tarih'].max()
        secilen_baslangic_tarihi = st.date_input("Başlangıç", min_tarih, min_value=min_tarih, max_value=max_tarih)
        secilen_bitis_tarihi = st.date_input("Bitiş", max_tarih, min_value=min_tarih, max_value=max_tarih)
    with col2:
        segment_listesi = ['Tümü'] + sonuclar_df['Segment'].unique().tolist()
        secilen_segmentler = st.multiselect("Segment Filtresi", segment_listesi, default=['Tümü'])

# Veriyi Filtrele
donem_df = temiz_df[
    (temiz_df['Tarih'].dt.date >= secilen_baslangic_tarihi) & 
    (temiz_df['Tarih'].dt.date <= secilen_bitis_tarihi)
]

# --- KPI KARTLARI ---
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

ciro = donem_df['ToplamTutar'].sum()
kar = donem_df['NetKar'].sum() if 'NetKar' in donem_df.columns else 0
islem = len(donem_df)
musteri = donem_df['MusteriID'].nunique()

col1.metric("Dönemsel Ciro", f"{ciro:,.0f} €")
col2.metric("Net Kar", f"{kar:,.0f} €")
col3.metric("İşlem Sayısı", f"{islem:,}")
col4.metric("Aktif Müşteri", f"{musteri}")

# --- AI ÖZET ---
st.markdown("---")
st.subheader("🤖 Yapay Zeka Özeti")
st.info(otomatik_ozet_uret(donem_df))

# --- GRAFİKLER ---
st.markdown("---")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Satış Trendi")
    aylik_satislar = genel_satis_trendi_hazirla(donem_df)
    fig_trend = px.area(aylik_satislar, x='ds', y='y', title='Aylık Ciro Gelişimi', labels={'ds': 'Tarih', 'y': 'Ciro'})
    st.plotly_chart(fig_trend, use_container_width=True)

with col_g2:
    st.subheader("Segment Dağılımı")
    segment_dagilimi = sonuclar_df[sonuclar_df.index.isin(donem_df['MusteriID'])]['Segment'].value_counts()
    fig_pie = px.pie(values=segment_dagilimi.values, names=segment_dagilimi.index, title="Aktif Müşterilerin Segment Dağılımı", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

# --- DETAY TABLOSU ---
st.markdown("---")
st.subheader("Segment Bazlı Müşteri Listesi")

tab_listesi = sorted(sonuclar_df['Segment'].unique())
tabs = st.tabs(tab_listesi)

profil_anomalileri = st.session_state.get('profil_anomalileri', [])
davranissal_anomaliler = st.session_state.get('davranissal_anomaliler', [])

for i, segment in enumerate(tab_listesi):
    with tabs[i]:
        # O segmentteki ve filtrelenen dönemdeki müşteriler
        seg_musteriler = sonuclar_df[
            (sonuclar_df['Segment'] == segment) & 
            (sonuclar_df.index.isin(donem_df['MusteriID']))
        ].copy()
        
        if seg_musteriler.empty:
            st.info("Bu filtreye uygun kayıt bulunamadı.")
        else:
            # Anomali İkonu Ekleme
            seg_musteriler['Durum'] = seg_musteriler.index.map(
                lambda x: "⚠️ Riskli" if (x in profil_anomalileri or x in davranissal_anomaliler) else "✅ Normal"
            )
            
            st.dataframe(
                seg_musteriler[['Durum', 'MusteriAdi', 'MPS', 'CLV_Net_Kar', 'Churn_Olasiligi', 'Recency', 'Monetary']]
                .sort_values('Monetary', ascending=False)
                .style.format({
                    'MPS': '{:.0f}', 'CLV_Net_Kar': '{:,.0f} €', 'Churn_Olasiligi': '{:.1%}',
                    'Recency': '{:.0f} Gün', 'Monetary': '{:,.0f} €'
                })
                .background_gradient(cmap='Greens', subset=['Monetary'])
            )