# pages/16_⏱️Gelecek_Satin_Alma.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar
st.set_page_config(page_title="Gelecek Satın Alma Tahmini", layout="wide")
make_sidebar()
yetki_kontrol("Gelecek Satın Alma Tahmini")

@st.cache_data
def veriyi_getir():
    # data_handler tek değer döndürüyor
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except Exception as e:
    st.error(f"Veri hatası: {e}")
    st.stop()

st.title("⏱️ Gelecek Satın Alma Zamanı Tahmini")
st.markdown("""
Bu modül, her bir müşterinin **alışveriş sıklığı alışkanlıklarını** analiz ederek, bir sonraki alışverişi **hangi tarihte** yapmasının beklendiğini hesaplar.
* **Gecikenler (Risk):** Tahmin edilen tarihi geçirmiş olanlar.
* **Beklenenler (Fırsat):** Yakın zamanda gelmesi beklenenler.
""")

# --- ANALİTİK MOTORU: ZAMANLAMA TAHMİNİ ---
@st.cache_data
def tahmin_motorunu_calistir(df):
    # Bugünü veri setindeki en son tarih olarak kabul edelim (Simülasyon için)
    analiz_tarihi = df['Tarih'].max()
    
    # En az 2 işlemi olan müşterileri al (Davranış analizi için tekrar şart)
    islem_sayilari = df['MusteriID'].value_counts()
    tekrar_eden_musteriler = islem_sayilari[islem_sayilari >= 2].index
    
    df_aktif = df[df['MusteriID'].isin(tekrar_eden_musteriler)].copy()
    df_aktif = df_aktif.sort_values(['MusteriID', 'Tarih'])
    
    # Bir önceki alışveriş tarihini bul
    df_aktif['Onceki_Tarih'] = df_aktif.groupby('MusteriID')['Tarih'].shift(1)
    
    # İki alışveriş arasındaki gün farkını (Gap) bul
    df_aktif['Gun_Farki'] = (df_aktif['Tarih'] - df_aktif['Onceki_Tarih']).dt.days
    
    # Müşteri bazında istatistikleri hesapla
    musteri_ozet = df_aktif.groupby('MusteriID').agg(
        Son_Alisveris=('Tarih', 'max'),
        Ortalama_Aralik=('Gun_Farki', 'median'), # Median outlier'lardan daha az etkilenir
        Standart_Sapma=('Gun_Farki', 'std'),
        Ortalama_Sepet=('ToplamTutar', 'mean'),
        Toplam_Ciro=('ToplamTutar', 'sum'),
        Islem_Sayisi=('Tarih', 'count')
    ).reset_index()
    
    # Standart sapması NaN olanları (sadece 2 işlemi olanlar) 0 yap
    musteri_ozet['Standart_Sapma'] = musteri_ozet['Standart_Sapma'].fillna(0)
    
    # --- TAHMİN HESAPLAMA ---
    # Beklenen Tarih = Son Alışveriş + Ortalama Aralık
    musteri_ozet['Beklenen_Tarih'] = musteri_ozet['Son_Alisveris'] + pd.to_timedelta(musteri_ozet['Ortalama_Aralik'], unit='D')
    
    # Gecikme Durumu (Bugüne göre)
    # Pozitif değer: Gecikmiş (Risk), Negatif değer: Daha vakti var
    musteri_ozet['Gecikme_Gun'] = (analiz_tarihi - musteri_ozet['Beklenen_Tarih']).dt.days
    
    # Güven Skoru: Standart sapma ne kadar düşükse, müşteri o kadar düzenlidir (Robot gibidir)
    # Basit bir skorlama: Düzenlilik Katsayısı
    # Eğer Std=0 ise (çok düzenli), Skor=100. Std arttıkça skor düşer.
    musteri_ozet['Tahmin_Guveni'] = np.where(
        musteri_ozet['Standart_Sapma'] == 0, 100, 
        100 / (1 + (musteri_ozet['Standart_Sapma'] / musteri_ozet['Ortalama_Aralik']))
    )
    
    return musteri_ozet, analiz_tarihi

with st.spinner("Müşteri alışkanlıkları ve zamanlamalar hesaplanıyor..."):
    tahmin_df, analiz_tarihi = tahmin_motorunu_calistir(df)

if tahmin_df.empty:
    st.warning("Tahmin yapabilmek için en az 2 kez alışveriş yapmış yeterli sayıda müşteri bulunamadı.")
    st.stop()

# --- SEGMENTASYON ---
def durum_etiketle(gecikme):
    if gecikme > 30: return "🚨 Kritik Gecikme (Churn Riski)"
    elif gecikme > 7: return "⚠️ Gecikmiş (Dikkat)"
    elif gecikme >= -7: return "📅 Eli Kulağında (Bu Hafta Bekleniyor)"
    else: return "✅ Güvende (Daha Vakti Var)"

tahmin_df['Durum'] = tahmin_df['Gecikme_Gun'].apply(durum_etiketle)

# --- KPI KARTLARI ---
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

# Gelecek 30 günde beklenen ciro (Basit projeksiyon)
gelecek_30_gun = tahmin_df[
    (tahmin_df['Beklenen_Tarih'] > analiz_tarihi) & 
    (tahmin_df['Beklenen_Tarih'] <= analiz_tarihi + timedelta(days=30))
]
beklenen_ciro = gelecek_30_gun['Ortalama_Sepet'].sum()

# Gecikmiş müşterilerin risk altındaki cirosu (Yıllık ortalama değerlerine göre risk)
riskli_musteriler = tahmin_df[tahmin_df['Gecikme_Gun'] > 7]
riskli_ciro_potansiyeli = riskli_musteriler['Ortalama_Sepet'].sum()

col1.metric("Analiz Tarihi (Bugün)", analiz_tarihi.strftime('%d-%m-%Y'))
col2.metric("Önümüzdeki 30 Günde Beklenen Ciro", f"{beklenen_ciro:,.0f} €", help="Günü gelen müşterilerin ortalama sepetlerine göre tahmini ciro.")
col3.metric("Bu Ay Beklenen Müşteri Sayısı", f"{len(gelecek_30_gun)}")
col4.metric("Risk Altındaki Ciro (Gecikenler)", f"{riskli_ciro_potansiyeli:,.0f} €", delta_color="inverse", help="Alışveriş periyodunu geçirmiş müşterilerin potansiyel cirosu.")

st.markdown("---")

# --- TABLOLAR VE AKSİYON LİSTELERİ ---
tab1, tab2 = st.tabs(["🚨 Acil Aksiyon Listesi (Gecikenler)", "📅 Gelecek Takvimi (Beklenenler)"])

with tab1:
    st.header("Geciken ve Riskli Müşteriler")
    st.markdown("Bu müşteriler normal alışveriş döngülerini aştılar. **Hemen aranmalı veya e-posta atılmalı.**")
    
    filtre_risk = riskli_musteriler.sort_values('Gecikme_Gun', ascending=False)
    
    st.dataframe(
        filtre_risk[['MusteriID', 'Durum', 'Gecikme_Gun', 'Ortalama_Aralik', 'Son_Alisveris', 'Ortalama_Sepet', 'Tahmin_Guveni']]
        .rename(columns={'Gecikme_Gun': 'Kaç Gün Gecikti?', 'Ortalama_Aralik': 'Normalde Kaç Günde Bir Gelir?', 'Ortalama_Sepet': 'Tahmini Sepet Tutarı'})
        .head(100) # Performans için ilk 100
        .style.format({
            'Kaç Gün Gecikti?': '{:.0f} gün',
            'Normalde Kaç Günde Bir Gelir?': '{:.0f} gün',
            'Tahmini Sepet Tutarı': '{:,.2f} €',
            'Son_Alisveris': lambda x: x.strftime('%d-%m-%Y'),
            'Tahmin_Guveni': '{:.0f}/100'
        })
        .background_gradient(cmap='Reds', subset=['Kaç Gün Gecikti?'])
    )

with tab2:
    st.header("Yakında Gelmesi Beklenenler")
    st.markdown("Bu müşterilerin alışveriş zamanı yaklaşıyor. Kendilerini hatırlatmak için iyi bir zaman.")
    
    filtre_gelecek = tahmin_df[tahmin_df['Gecikme_Gun'] <= 0].sort_values('Beklenen_Tarih', ascending=True)
    
    st.dataframe(
        filtre_gelecek[['MusteriID', 'Durum', 'Beklenen_Tarih', 'Ortalama_Aralik', 'Ortalama_Sepet', 'Tahmin_Guveni']]
        .rename(columns={'Beklenen_Tarih': 'Tahmini Gelis Tarihi', 'Ortalama_Sepet': 'Beklenen Tutar'})
        .head(100)
        .style.format({
            'Tahmini Gelis Tarihi': lambda x: x.strftime('%d-%m-%Y'),
            'Ortalama_Aralik': '{:.0f} gün',
            'Beklenen Tutar': '{:,.2f} €',
            'Tahmin_Guveni': '{:.0f}/100'
        })
        .background_gradient(cmap='Greens', subset=['Tahmin_Guveni'])
    )

# --- GÖRSELLEŞTİRME ---
st.markdown("---")
st.subheader("📊 Müşteri Sadakat Analizi: Sıklık vs Düzenlilik")
st.markdown("Sağ üst köşe: **Hem sık hem düzenli** (En değerli robot müşteriler).")

fig_scatter = px.scatter(
    tahmin_df, 
    x="Ortalama_Aralik", 
    y="Tahmin_Guveni", 
    size="Ortalama_Sepet", 
    color="Durum",
    hover_name="MusteriID",
    title="Müşteri Davranış Haritası",
    labels={"Ortalama_Aralik": "Ortalama Alışveriş Aralığı (Gün)", "Tahmin_Guveni": "Davranış Düzenliliği (Güven Skoru)"},
    color_discrete_map={
        "🚨 Kritik Gecikme (Churn Riski)": "red",
        "⚠️ Gecikmiş (Dikkat)": "orange",
        "📅 Eli Kulağında (Bu Hafta Bekleniyor)": "blue",
        "✅ Güvende (Daha Vakti Var)": "green"
    }
)
st.plotly_chart(fig_scatter, use_container_width=True)