# pages/29_🎨Benim_Panom.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import rfm_skorlarini_hesapla, musterileri_segmentle, churn_tahmin_modeli_olustur, clv_hesapla
from auth_manager import yetki_kontrol
from navigation import make_sidebar
st.set_page_config(page_title="Kişisel Dashboard", layout="wide")
make_sidebar()
yetki_kontrol("Kişisel Dashboard")

@st.cache_data
def veriyi_getir_ve_analiz_et():
    # 1. Ham Veri
    df = veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')
    
    # 2. Müşteri Analitiği (RFM, Segment, Churn, CLV)
    # Panoda müşteri grafikleri de göstermek için bunları hesaplamamız lazım
    rfm = rfm_skorlarini_hesapla(df)
    seg = musterileri_segmentle(rfm)
    churn_df, _, _, _, _, _ = churn_tahmin_modeli_olustur(seg)
    sonuclar = clv_hesapla(churn_df)
    
    return df, sonuclar

try:
    df, sonuclar_df = veriyi_getir_ve_analiz_et()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🎨 Benim Panom (Executive Dashboard)")
st.markdown("Aşağıdaki menüden, günlük takibini yapmak istediğiniz grafikleri seçerek **kendi yönetim ekranınızı** oluşturun.")

# ==========================================
# 🧩 WIDGET KÜTÜPHANESİ (GRAFİK MOTORLARI)
# ==========================================

# --- FİNANSAL WIDGETLAR ---
def widget_kpi_ozet(df, sonuclar_df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Ciro", f"{df['ToplamTutar'].sum():,.0f} €")
    c2.metric("Toplam Net Kar", f"{df['NetKar'].sum():,.0f} €")
    c3.metric("Aktif Müşteri", f"{df['MusteriID'].nunique()}")
    c4.metric("Ort. Sepet Tutarı", f"{df['ToplamTutar'].mean():,.0f} €")

def widget_satis_trendi(df, sonuclar_df):
    monthly = df.set_index('Tarih').resample('M')['ToplamTutar'].sum().reset_index()
    fig = px.area(monthly, x='Tarih', y='ToplamTutar', title="Aylık Satış Trendi", markers=True)
    st.plotly_chart(fig, use_container_width=True)

def widget_kar_marji_trendi(df, sonuclar_df):
    # Aylık Kar ve Ciro
    monthly = df.set_index('Tarih').resample('M').agg({'ToplamTutar':'sum', 'NetKar':'sum'}).reset_index()
    monthly['KarMarji'] = (monthly['NetKar'] / monthly['ToplamTutar']) * 100
    
    fig = px.line(monthly, x='Tarih', y='KarMarji', title="Aylık Kar Marjı Trendi (%)", markers=True, color_discrete_sequence=['green'])
    fig.add_hline(y=monthly['KarMarji'].mean(), line_dash="dash", line_color="gray", annotation_text="Ortalama")
    st.plotly_chart(fig, use_container_width=True)

# --- ÜRÜN VE STOK WIDGETLARI ---
def widget_top_urunler(df, sonuclar_df):
    top = df.groupby('UrunKodu')['ToplamTutar'].sum().nlargest(10).reset_index()
    fig = px.bar(top, x='ToplamTutar', y='UrunKodu', orientation='h', title="En Çok Satan 10 Ürün (Ciro)")
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def widget_kategori_dagilimi(df, sonuclar_df):
    if 'Kategori' in df.columns:
        cat = df.groupby('Kategori')['ToplamTutar'].sum().reset_index()
        fig = px.pie(cat, values='ToplamTutar', names='Kategori', title="Kategori Ciro Dağılımı", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kategori verisi yok.")

def widget_pareto_durumu(df, sonuclar_df):
    # Ürünlerin % kaçı cironun %80'ini yapıyor?
    ozet = df.groupby('UrunKodu')['ToplamTutar'].sum().sort_values(ascending=False).reset_index()
    ozet['Kumulatif'] = ozet['ToplamTutar'].cumsum()
    ozet['Oran'] = ozet['Kumulatif'] / ozet['ToplamTutar'].sum()
    a_sinifi = len(ozet[ozet['Oran'] <= 0.8])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = a_sinifi,
        title = {'text': "A Sınıfı (Kritik) Ürün Sayısı"},
        gauge = {'axis': {'range': [0, len(ozet)]}, 'bar': {'color': "darkred"}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# --- MÜŞTERİ WIDGETLARI ---
def widget_segment_dagilimi(df, sonuclar_df):
    seg_counts = sonuclar_df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'KisiSayisi']
    fig = px.bar(seg_counts, x='Segment', y='KisiSayisi', color='Segment', title="Müşteri Segment Dağılımı")
    st.plotly_chart(fig, use_container_width=True)

def widget_churn_riski(df, sonuclar_df):
    # Churn olasılığı %50'den büyük olanlar riskli
    riskli_sayi = len(sonuclar_df[sonuclar_df['Churn_Olasiligi'] > 0.5])
    guvenli_sayi = len(sonuclar_df) - riskli_sayi
    
    fig = px.pie(
        names=['Güvende', 'Churn Riski Yüksek'], 
        values=[guvenli_sayi, riskli_sayi],
        color_discrete_map={'Güvende':'#2ecc71', 'Churn Riski Yüksek':'#e74c3c'},
        title="Müşteri Tabanı Risk Durumu"
    )
    st.plotly_chart(fig, use_container_width=True)

def widget_clv_dagilimi(df, sonuclar_df):
    fig = px.histogram(sonuclar_df, x="CLV_Net_Kar", nbins=50, title="Müşteri Yaşam Boyu Değeri (CLV) Dağılımı")
    fig.update_layout(xaxis_title="CLV (€)", yaxis_title="Müşteri Sayısı")
    st.plotly_chart(fig, use_container_width=True)

# --- OPERASYONEL WIDGETLAR ---
def widget_gunluk_yogunluk(df, sonuclar_df):
    df['Gun'] = df['Tarih'].dt.day_name()
    gun_sirasi = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Türkçe mapping
    tr_gunler = {'Monday':'Pazartesi', 'Tuesday':'Salı', 'Wednesday':'Çarşamba', 'Thursday':'Perşembe', 'Friday':'Cuma', 'Saturday':'Cumartesi', 'Sunday':'Pazar'}
    df['GunTR'] = df['Gun'].map(tr_gunler)
    
    daily = df.groupby('GunTR')['ToplamTutar'].sum().reindex(list(tr_gunler.values())).reset_index()
    fig = px.bar(daily, x='GunTR', y='ToplamTutar', title="Haftanın Günlerine Göre Satış Yoğunluğu")
    st.plotly_chart(fig, use_container_width=True)

def widget_maliyet_trendi(df, sonuclar_df):
    if 'Maliyet' in df.columns:
        # --- DÜZELTME: Sütun yoksa anlık hesapla ---
        if 'ToplamMaliyet' not in df.columns:
            df['ToplamMaliyet'] = df['Miktar'] * df['Maliyet']
            
        monthly_cost = df.set_index('Tarih').resample('M')['ToplamMaliyet'].sum().reset_index()
        fig = px.line(monthly_cost, x='Tarih', y='ToplamMaliyet', title="Aylık Satınalma Maliyeti (Cash Outflow)", markers=True, color_discrete_sequence=['red'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Maliyet verisi yok.")

# ==========================================
# 🎛️ SIDEBAR SEÇİM MEKANİZMASI
# ==========================================

# Widget Sözlüğü (Kategori: {İsim: Fonksiyon})
widget_katalogu = {
    "Finansal": {
        "KPI Özeti (Kartlar)": widget_kpi_ozet,
        "Satış Trendi (Alan Grafiği)": widget_satis_trendi,
        "Kar Marjı Trendi (%)": widget_kar_marji_trendi,
    },
    "Ürün & Stok": {
        "Top 10 Ürün (Bar)": widget_top_urunler,
        "Kategori Dağılımı (Pasta)": widget_kategori_dagilimi,
        "Pareto Kritik Ürün (Gösterge)": widget_pareto_durumu,
    },
    "Müşteri (CRM)": {
        "Segment Dağılımı (Bar)": widget_segment_dagilimi,
        "Churn Riski (Pasta)": widget_churn_riski,
        "CLV Dağılımı (Histogram)": widget_clv_dagilimi,
    },
    "Operasyonel": {
        "Günlük Satış Yoğunluğu": widget_gunluk_yogunluk,
        "Satınalma Nakit Çıkışı": widget_maliyet_trendi,
    }
}

st.sidebar.header("🛠️ Panonu Tasarla")
secilenler = []

# Her kategori için expander açıp seçim yaptıralım
for kategori, widgetlar in widget_katalogu.items():
    with st.sidebar.expander(f"{kategori} Analizleri", expanded=True):
        for widget_adi, widget_func in widgetlar.items():
            # Varsayılan olarak bazılarını seçili getir
            varsayilan = True if widget_adi in ["KPI Özeti (Kartlar)", "Satış Trendi (Alan Grafiği)", "Top 10 Ürün (Bar)"] else False
            if st.checkbox(widget_adi, value=varsayilan):
                secilenler.append((widget_adi, widget_func))

# ==========================================
# 🖼️ PANO YERLEŞİMİ
# ==========================================

if not secilenler:
    st.info("👈 Lütfen sol menüden en az bir analiz seçin.")
else:
    # 1. KPI Özeti her zaman en üstte ve tam genişlikte olsun (Eğer seçildiyse)
    # Listede KPI Özeti var mı kontrol et
    kpi_var = False
    for ad, func in secilenler:
        if ad == "KPI Özeti (Kartlar)":
            func(df, sonuclar_df)
            st.markdown("---")
            kpi_var = True
            break
    
    # KPI'ı tekrar çizmemek için listeden filtrele
    kalan_widgetlar = [w for w in secilenler if w[0] != "KPI Özeti (Kartlar)"]
    
    # 2. Diğer Grafikler (2'li Izgara Sistemi)
    col1, col2 = st.columns(2)
    
    for i, (ad, func) in enumerate(kalan_widgetlar):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"##### {ad}")
            func(df, sonuclar_df)
            st.markdown("---")