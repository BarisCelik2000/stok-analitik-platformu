# pages/27_📅Stok_Yaslandirma.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar
import numpy as np
# Yardım içeriği varsa ekle, yoksa geç
try:
    from help_content import yardim_goster
    yardim_goster("Stok Yaşlandırma") 
except:
    pass

st.set_page_config(page_title="Stok Yaşlandırma Raporu", layout="wide")
make_sidebar()
yetki_kontrol("Stok Yaşlandırma Raporu")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("📅 Stok Yaşlandırma Raporu (Inventory Aging)")
st.markdown("""
Deponuzdaki ürünlerin **hareketsizlik sürelerini** analiz eder.
Nakit paranızın ne kadarının "yavaş dönen" veya "ölü" stoklarda bağlı olduğunu gösterir.
""")

# --- VERİ HAZIRLIĞI ---
# Analiz tarihi (Bugün)
analiz_tarihi = df['Tarih'].max()

# Ürün bazında son hareket tarihini bul
stok_yas_df = df.groupby('UrunKodu').agg(
    SonSatisTarihi=('Tarih', 'max'),
    ToplamSatisAdedi=('Miktar', 'sum'),
    BirimMaliyet=('Maliyet', 'mean') if 'Maliyet' in df.columns else ('BirimFiyat', lambda x: x.mean() * 0.75)
).reset_index()

# Hareketsizlik Süresi (Gün)
stok_yas_df['HareketsizGun'] = (analiz_tarihi - stok_yas_df['SonSatisTarihi']).dt.days

# Tahmini Mevcut Stok (Simülasyon: Toplam satışın %10'u kadar stok var varsayalım)
# Gerçek ERP verisinde bu sütun "MevcutStok" olarak doğrudan gelir.
stok_yas_df['TahminiStok'] = (stok_yas_df['ToplamSatisAdedi'] * 0.10).apply(np.ceil)
stok_yas_df['StokDegeri'] = stok_yas_df['TahminiStok'] * stok_yas_df['BirimMaliyet']

# Yaşlandırma Kovaları (Buckets)
def yas_kovasi(gun):
    if gun <= 30: return "0-30 Gün (Taze)"
    elif gun <= 60: return "31-60 Gün (Yavaş)"
    elif gun <= 90: return "61-90 Gün (Riskli)"
    else: return "90+ Gün (Ölü Stok)"

stok_yas_df['YasGrubu'] = stok_yas_df['HareketsizGun'].apply(yas_kovasi)

# Sıralama için kategori tipi yapalım
kategoriler = ["0-30 Gün (Taze)", "31-60 Gün (Yavaş)", "61-90 Gün (Riskli)", "90+ Gün (Ölü Stok)"]
stok_yas_df['YasGrubu'] = pd.Categorical(stok_yas_df['YasGrubu'], categories=kategoriler, ordered=True)

# --- KPI KARTLARI ---
toplam_stok_degeri = stok_yas_df['StokDegeri'].sum()
olu_stok_degeri = stok_yas_df[stok_yas_df['YasGrubu'] == "90+ Gün (Ölü Stok)"]['StokDegeri'].sum()
olu_stok_orani = (olu_stok_degeri / toplam_stok_degeri) * 100

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Stok Değeri (Tahmini)", f"{toplam_stok_degeri:,.0f} €")
col2.metric("Ölü Stok Değeri (90+ Gün)", f"{olu_stok_degeri:,.0f} €", delta_color="inverse")
col3.metric("Ölü Stok Oranı", f"%{olu_stok_orani:.1f}", delta_color="inverse")

# --- GÖRSELLEŞTİRME ---
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    st.subheader("Stok Yaş Dağılımı (Tutar Bazlı)")
    
    yas_ozet = stok_yas_df.groupby('YasGrubu')['StokDegeri'].sum().reset_index()
    
    fig_pie = px.pie(
        yas_ozet, 
        values='StokDegeri', 
        names='YasGrubu',
        color='YasGrubu',
        color_discrete_map={
            "0-30 Gün (Taze)": "#2ecc71",
            "31-60 Gün (Yavaş)": "#f1c40f",
            "61-90 Gün (Riskli)": "#e67e22",
            "90+ Gün (Ölü Stok)": "#e74c3c"
        },
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c_chart2:
    st.subheader("Yaş Gruplarına Göre Ürün Sayısı")
    count_ozet = stok_yas_df.groupby('YasGrubu')['UrunKodu'].count().reset_index()
    
    fig_bar = px.bar(
        count_ozet, 
        x='YasGrubu', 
        y='UrunKodu',
        text='UrunKodu',
        color='YasGrubu',
        color_discrete_map={
            "0-30 Gün (Taze)": "#2ecc71",
            "31-60 Gün (Yavaş)": "#f1c40f",
            "61-90 Gün (Riskli)": "#e67e22",
            "90+ Gün (Ölü Stok)": "#e74c3c"
        }
    )
    fig_bar.update_layout(showlegend=False, yaxis_title="Ürün Çeşidi Sayısı")
    st.plotly_chart(fig_bar, use_container_width=True)

# --- DETAY TABLOSU ---
st.markdown("---")
st.subheader("📋 Riskli Ürünler Listesi (60 Gün ve Üzeri)")

riskli_liste = stok_yas_df[stok_yas_df['HareketsizGun'] > 60].sort_values('HareketsizGun', ascending=False)

if not riskli_liste.empty:
    st.dataframe(
        riskli_liste[['UrunKodu', 'YasGrubu', 'HareketsizGun', 'SonSatisTarihi', 'TahminiStok', 'StokDegeri']]
        .style.format({
            'SonSatisTarihi': lambda x: x.strftime('%d-%m-%Y'),
            'TahminiStok': '{:,.0f}',
            'StokDegeri': '{:,.2f} €'
        })
        .background_gradient(cmap='Reds', subset=['HareketsizGun'])
    )
    
    # Excel İndirme (Kütüphaneyi yüklediğiniz için artık çalışır)
    try:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            riskli_liste.to_excel(writer, sheet_name='Riskli Stoklar', index=False)
            
        st.download_button(
            label="📥 Riskli Stok Listesini İndir (Excel)",
            data=buffer.getvalue(),
            file_name="stok_yaslandirma_raporu.xlsx",
            mime="application/vnd.ms-excel"
        )
    except Exception as e:
        st.warning("Excel indirme butonu oluşturulamadı (xlsxwriter eksik olabilir).")
else:
    st.success("60 günden eski hareketsiz stok bulunmuyor.")