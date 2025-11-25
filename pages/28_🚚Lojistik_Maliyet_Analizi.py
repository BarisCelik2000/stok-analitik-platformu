# pages/28_🚚Lojistik_Maliyet_Analizi.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
try:
    from help_content import yardim_goster
    yardim_goster("Lojistik Maliyet Analizi") 
except:
    pass

st.set_page_config(page_title="Lojistik Maliyet Analizi", layout="wide")
make_sidebar()
yetki_kontrol("Lojistik Maliyet Analizi")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🚚 Lojistik Maliyet Analizi (Landed Cost)")
st.markdown("""
Ürünlerin **Net İndirilmiş Maliyetini (Landed Cost)** hesaplayarak gerçek karlılığı görün.
Fatura maliyetinin üzerine binen nakliye, gümrük ve operasyonel giderleri simüle edin.
""")

# --- VERİ HAZIRLIĞI ---
if 'Maliyet' not in df.columns:
    df['Maliyet'] = df['BirimFiyat'] * 0.75

# --- SİMÜLASYON PARAMETRELERİ ---
with st.sidebar:
    st.header("⚙️ Lojistik Gider Varsayımları")
    st.info("Veri setinde lojistik kalemleri olmadığı için bu oranları simüle ediyoruz.")
    
    navlun_orani = st.slider("Ortalama Navlun (Nakliye) %", 0, 20, 5, help="Ürün maliyetinin % kaçı nakliyeye gidiyor?")
    gumruk_orani = st.slider("Gümrük ve Vergi %", 0, 30, 10, help="İthalat vergileri ve gümrük masrafları.")
    ellecleme_orani = st.slider("Depo ve Elleçleme %", 0, 10, 2, help="İndirme, bindirme ve depolama maliyeti.")

# --- HESAPLAMALAR ---
# Lojistik maliyetlerini ürün bazında hesapla
df_analiz = df.groupby('UrunKodu').agg(
    OrtalamaBirimFiyat=('BirimFiyat', 'mean'),
    OrtalamaMaliyet=('Maliyet', 'mean'), # Fabrika Çıkış (EXW/FOB)
    ToplamSatisAdedi=('Miktar', 'sum')
).reset_index()

# Landed Cost Hesaplama
# Formül: Maliyet * (1 + (Tüm Oranlar/100))
toplam_ek_oran = (navlun_orani + gumruk_orani + ellecleme_orani) / 100

df_analiz['LojistikMaliyeti'] = df_analiz['OrtalamaMaliyet'] * (navlun_orani / 100)
df_analiz['GumrukMaliyeti'] = df_analiz['OrtalamaMaliyet'] * (gumruk_orani / 100)
df_analiz['ElleclemeMaliyeti'] = df_analiz['OrtalamaMaliyet'] * (ellecleme_orani / 100)

df_analiz['LandedCost'] = df_analiz['OrtalamaMaliyet'] * (1 + toplam_ek_oran)

# Karlılık Karşılaştırması
df_analiz['BrutKarMarji'] = (df_analiz['OrtalamaBirimFiyat'] - df_analiz['OrtalamaMaliyet']) / df_analiz['OrtalamaBirimFiyat']
df_analiz['NetKarMarji'] = (df_analiz['OrtalamaBirimFiyat'] - df_analiz['LandedCost']) / df_analiz['OrtalamaBirimFiyat']

# Riskli Ürünler (Lojistik sonrası zarar edenler)
zarar_edenler = df_analiz[df_analiz['NetKarMarji'] <= 0].sort_values('NetKarMarji')

# --- GÖRSELLEŞTİRME VE KPI ---
kpi1, kpi2, kpi3 = st.columns(3)

ort_exw = df_analiz['OrtalamaMaliyet'].mean()
ort_landed = df_analiz['LandedCost'].mean()
maliyet_artisi = ort_landed - ort_exw

kpi1.metric("Ortalama Fabrika Maliyeti", f"{ort_exw:,.2f} €")
kpi2.metric("Ortalama İndirilmiş Maliyet (Landed)", f"{ort_landed:,.2f} €", delta=f"-{maliyet_artisi:.2f} € Ek Gider", delta_color="inverse")
kpi3.metric("Riskli Ürün Sayısı", len(zarar_edenler), help="Lojistik maliyetleri eklenince zarar eden ürünler.")

st.markdown("---")

# --- WATERFALL CHART (MALİYET ŞELALESİ) ---
st.subheader("💰 Birim Maliyet Kırılımı (Waterfall)")
st.markdown("Bir ürünün maliyetinin depoya girene kadar nasıl katlandığını inceleyin.")

secilen_urun = st.selectbox("Detaylı incelemek için ürün seçin:", df_analiz['UrunKodu'].unique())

if secilen_urun:
    urun_datasi = df_analiz[df_analiz['UrunKodu'] == secilen_urun].iloc[0]
    
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Maliyet Yapısı",
        orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Fabrika Maliyeti", "Navlun", "Gümrük", "Elleçleme", "NET MALİYET (Landed)"],
        textposition = "outside",
        text = [f"{urun_datasi['OrtalamaMaliyet']:.2f}€", 
                f"{urun_datasi['LojistikMaliyeti']:.2f}€", 
                f"{urun_datasi['GumrukMaliyeti']:.2f}€", 
                f"{urun_datasi['ElleclemeMaliyeti']:.2f}€", 
                f"{urun_datasi['LandedCost']:.2f}€"],
        y = [urun_datasi['OrtalamaMaliyet'], 
             urun_datasi['LojistikMaliyeti'], 
             urun_datasi['GumrukMaliyeti'], 
             urun_datasi['ElleclemeMaliyeti'], 
             0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig_waterfall.update_layout(
        title = f"'{secilen_urun}' Maliyet Bileşenleri",
        showlegend = False
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Karlılık Uyarısı
    st.info(f"""
    📊 **Karlılık Analizi:**
    * **Satış Fiyatı:** {urun_datasi['OrtalamaBirimFiyat']:.2f} €
    * **Kağıt Üstünde Kar:** {urun_datasi['OrtalamaBirimFiyat'] - urun_datasi['OrtalamaMaliyet']:.2f} € (Brüt)
    * **Gerçek Kar:** {urun_datasi['OrtalamaBirimFiyat'] - urun_datasi['LandedCost']:.2f} € (Net Operasyonel)
    """)
    
    if urun_datasi['NetKarMarji'] < 0:
        st.error("🚨 DİKKAT: Lojistik maliyetleri eklendiğinde bu üründen zarar ediyorsunuz!")

# --- RİSK TABLOSU ---
st.markdown("---")
st.subheader("📉 Gizli Zarar Eden Ürünler (Hidden Loss)")
st.markdown("Brüt karı pozitif görünen ancak lojistik masrafları düşünce zarar yazan ürünler.")

if not zarar_edenler.empty:
    st.dataframe(
        zarar_edenler[['UrunKodu', 'OrtalamaBirimFiyat', 'OrtalamaMaliyet', 'LandedCost', 'BrutKarMarji', 'NetKarMarji']]
        .style.format({
            'OrtalamaBirimFiyat': '{:.2f} €',
            'OrtalamaMaliyet': '{:.2f} €',
            'LandedCost': '{:.2f} €',
            'BrutKarMarji': '{:.1%}',
            'NetKarMarji': '{:.1%}'
        })
        .background_gradient(cmap='Reds_r', subset=['NetKarMarji'])
    )
else:
    st.success("Harika! Seçilen lojistik parametrelerine göre zarar eden ürününüz bulunmuyor.")