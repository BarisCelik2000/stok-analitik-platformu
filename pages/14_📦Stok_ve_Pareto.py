# pages/14_📦Stok_ve_Pareto_Analizi.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from auth_manager import yetki_kontrol
# Merkezi veri yükleme fonksiyonu (Düzeltilmiş haliyle)
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar
st.set_page_config(page_title="Stok ve Pareto Analizi", layout="wide")
make_sidebar()
yetki_kontrol("Stok ve Pareto Analizi")

@st.cache_data
def veriyi_getir():
    dosya_adi = 'satis_verileri_guncellenmis.json'
    # Artık tek değer dönüyor, hata almayacaksınız:
    df = veriyi_yukle_ve_temizle(dosya_adi)
    return df

try:
    df = veriyi_getir()
except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

st.title("📦 Stok Dağılımı ve Pareto (ABC) Analizi")
st.markdown("Bu modül, ürün portföyünüzün verimliliğini, 80/20 kuralını ve satış hızı düşen 'Ölü Stok' adaylarını analiz eder.")

# --- ANALİZ HAZIRLIĞI ---
# Kategori bazlı özet
if 'Kategori' in df.columns:
    ozet_df = df.groupby(['Kategori', 'UrunKodu']).agg(
        ToplamCiro=('ToplamTutar', 'sum'),
        ToplamAdet=('Miktar', 'sum'),
        SonSatisTarihi=('Tarih', 'max')
    ).reset_index()
else:
    ozet_df = df.groupby('UrunKodu').agg(
        ToplamCiro=('ToplamTutar', 'sum'),
        ToplamAdet=('Miktar', 'sum'),
        SonSatisTarihi=('Tarih', 'max')
    ).reset_index()
    ozet_df['Kategori'] = 'Genel'

tab1, tab2, tab3 = st.tabs(["🌳 Ürün Ağaç Haritası (Treemap)", "⚖️ Pareto (80/20) Analizi", "💀 Ölü Stok Riski"])

# --- TAB 1: TREEMAP ---
with tab1:
    st.header("Ürün ve Kategori Hiyerarşisi")
    st.markdown("Kutuların büyüklüğü **Ciro**, rengi ise **Satış Adedi** yoğunluğunu temsil eder.")
    
    # Treemap interaktif olduğu için çok veride yavaşlayabilir, top 500 ürünü alalım
    top_urunler = ozet_df.nlargest(500, 'ToplamCiro')
    
    fig_tree = px.treemap(
        top_urunler, 
        path=[px.Constant("Tüm Ürünler"), 'Kategori', 'UrunKodu'], 
        values='ToplamCiro',
        color='ToplamAdet',
        color_continuous_scale='Viridis',
        title="Satış Dağılımı Ağaç Haritası (Ciro Bazlı)",
        hover_data=['ToplamCiro', 'ToplamAdet']
    )
    fig_tree.update_traces(root_color="lightgrey")
    fig_tree.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_tree, use_container_width=True)
    st.info("💡 **İpucu:** Kutuların üzerine tıklayarak kategorilerin içine girebilir (zoom in) ve ürün detaylarını görebilirsiniz.")

# --- TAB 2: PARETO ANALİZİ ---
with tab2:
    st.header("Pareto Prensibi (80/20 Kuralı)")
    st.markdown("Genellikle cironun %80'i, ürünlerin sadece %20'sinden gelir. Bu ürünler sizin için **kritik öneme** sahiptir.")
    
    # Pareto Hesaplaması
    pareto_df = ozet_df.groupby('UrunKodu')['ToplamCiro'].sum().reset_index()
    pareto_df = pareto_df.sort_values(by='ToplamCiro', ascending=False)
    pareto_df['KümülatifCiro'] = pareto_df['ToplamCiro'].cumsum()
    pareto_df['KümülatifYuzde'] = 100 * pareto_df['KümülatifCiro'] / pareto_df['ToplamCiro'].sum()
    
    # Ürünleri sınıflandır
    def abc_sinifi(yuzde):
        if yuzde <= 80: return 'A (Çok Kritik)'
        elif yuzde <= 95: return 'B (Önemli)'
        else: return 'C (Standart)'
        
    pareto_df['Sinif'] = pareto_df['KümülatifYuzde'].apply(abc_sinifi)
    
    # Görselleştirme
    a_sinifi_sayisi = len(pareto_df[pareto_df['Sinif'] == 'A (Çok Kritik)'])
    toplam_urun = len(pareto_df)
    a_sinifi_orani = (a_sinifi_sayisi / toplam_urun) * 100
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Toplam Ürün Çeşidi", toplam_urun)
        st.metric("Cironun %80'ini Oluşturan Ürün Sayısı", a_sinifi_sayisi)
        st.metric("Ürün Portföyündeki Oranı", f"%{a_sinifi_orani:.1f}")
        
        st.warning(f"⚠️ Ürünlerinizin sadece **%{a_sinifi_orani:.1f}**'lik kısmı, cironuzun **%80**'ini taşıyor. Bu ürünlerin stoğu asla tükenmemeli!")

    with col2:
        fig_pareto = px.bar(
            pareto_df.head(50), 
            x='UrunKodu', 
            y='ToplamCiro', 
            color='Sinif',
            title='En Çok Ciro Getiren Top 50 Ürün ve Pareto Sınıfları',
            color_discrete_map={'A (Çok Kritik)': '#EF553B', 'B (Önemli)': '#FFA15A', 'C (Standart)': '#636EFA'}
        )
        # Kümülatif çizgi ekle
        fig_pareto.add_trace(
            go.Scatter(
                x=pareto_df.head(50)['UrunKodu'], 
                y=pareto_df.head(50)['KümülatifCiro'], 
                mode='lines', 
                name='Kümülatif Ciro', 
                yaxis='y2',
                line=dict(color='black', width=2, dash='dot')
            )
        )
        fig_pareto.update_layout(
            yaxis2=dict(title='Kümülatif Ciro', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.6, y=0.9)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with st.expander("A Sınıfı (En Değerli) Ürün Listesini İndir"):
        a_sinifi_df = pareto_df[pareto_df['Sinif'] == 'A (Çok Kritik)']
        st.dataframe(a_sinifi_df)

# --- TAB 3: ÖLÜ STOK ANALİZİ ---
with tab3:
    st.header("Ölü Stok (Dead Stock) Riski Analizi")
    st.markdown("Uzun süredir satışı gerçekleşmeyen ürünleri tespit edin.")
    
    analiz_tarihi = df['Tarih'].max()
    ozet_df['SonSatisGunOnce'] = (analiz_tarihi - ozet_df['SonSatisTarihi']).dt.days
    
    esik_deger = st.slider("Kaç gündür satılmayan ürünler 'Riskli' sayılsın?", 30, 365, 90)
    
    riskli_stoklar = ozet_df[ozet_df['SonSatisGunOnce'] > esik_deger].sort_values('SonSatisGunOnce', ascending=False)
    
    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        st.error(f"🚨 **{len(riskli_stoklar)}** adet ürün {esik_deger} gündür hiç satılmadı!")
    with col_risk2:
        # Son satış tarihlerine göre histogram
        fig_hist = px.histogram(riskli_stoklar, x="SonSatisGunOnce", nbins=20, title="Riskli Ürünlerin Satışsız Geçen Gün Dağılımı")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Riskli Ürünler Listesi")
    st.dataframe(
        riskli_stoklar[['Kategori', 'UrunKodu', 'SonSatisTarihi', 'SonSatisGunOnce', 'ToplamCiro']]
        .style.format({'SonSatisTarihi': lambda x: x.strftime('%d-%m-%Y'), 'ToplamCiro': '{:,.2f} €'})
        .background_gradient(cmap='Reds', subset=['SonSatisGunOnce'])
    )