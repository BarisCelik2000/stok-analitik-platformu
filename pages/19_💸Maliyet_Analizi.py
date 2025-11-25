# pages/19_💸Maliyet_ve_Harcama.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
# Akıllı Rehber Entegrasyonu
try:
    from help_content import yardim_goster
    yardim_goster("Maliyet ve Harcama")
except:
    pass

st.set_page_config(page_title="Maliyet ve Harcama Analizi", layout="wide")
make_sidebar()
yetki_kontrol("Maliyet ve Harcama Analizi")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

# --- VERİ HAZIRLIĞI ---
# Maliyet sütunu yoksa, varsayımsal maliyet oluştur (Cironun %75'i)
if 'Maliyet' not in df.columns:
    df['Maliyet'] = df['BirimFiyat'] * 0.75
    df['ToplamMaliyet'] = df['ToplamTutar'] * 0.75
else:
    df['ToplamMaliyet'] = df['Miktar'] * df['Maliyet']

# Kar Marjı Hesaplama (Birim Bazlı)
# Marj % = (Fiyat - Maliyet) / Fiyat
df['BirimKar'] = df['BirimFiyat'] - df['Maliyet']
df['KarMarji'] = (df['BirimKar'] / df['BirimFiyat']) * 100

st.title("💸 Maliyet ve Harcama Analizi")
st.markdown("Tek tedarikçili yapıda maliyetlerinizi, harcama dağılımınızı ve ürün karlılıklarını analiz edin.")

# --- KPI KARTLARI ---
toplam_harcama = df['ToplamMaliyet'].sum()
toplam_ciro = df['ToplamTutar'].sum()
maliyet_orani = (toplam_harcama / toplam_ciro) * 100
ortalama_birim_maliyet = df['Maliyet'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Satınalma Maliyeti (COGS)", f"{toplam_harcama:,.0f} €", help="Satılan Malların Maliyeti")
col2.metric("Maliyetin Ciroya Oranı", f"%{maliyet_orani:.1f}", help="Cironun ne kadarı maliyete gidiyor? Düşük olması iyidir.")
col3.metric("Yönetilen Ürün (SKU)", f"{df['UrunKodu'].nunique()}")
col4.metric("Ortalama Birim Maliyet", f"{ortalama_birim_maliyet:,.2f} €")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📉 Harcama Analizi (Pareto)", "📊 Maliyet Trendleri (PPV)", "💎 Ürün Karlılık Matrisi"])

# --- TAB 1: HARCAMA ANALİZİ (SPEND ANALYSIS) ---
with tab1:
    st.header("Harcama Dağılımı ve Pareto")
    st.markdown("Bütçenizin büyük kısmı hangi ürünlere gidiyor?")
    
    col_spend1, col_spend2 = st.columns([2, 1])
    
    with col_spend1:
        # Kategori Bazlı Harcama (Treemap)
        if 'Kategori' in df.columns:
            path_list = ['Kategori', 'UrunKodu']
        else:
            path_list = ['UrunKodu']
            
        spend_tree = df.groupby(path_list)['ToplamMaliyet'].sum().reset_index()
        
        fig_tree = px.treemap(
            spend_tree,
            path=path_list,
            values='ToplamMaliyet',
            title="Kategori ve Ürün Bazlı Maliyet Dağılımı",
            color='ToplamMaliyet',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with col_spend2:
        # ABC Analizi (Maliyet Odaklı)
        st.subheader("Maliyet Pareto (ABC)")
        product_spend = df.groupby('UrunKodu')['ToplamMaliyet'].sum().sort_values(ascending=False).reset_index()
        product_spend['Kumulatif'] = product_spend['ToplamMaliyet'].cumsum()
        product_spend['Oran'] = product_spend['Kumulatif'] / product_spend['ToplamMaliyet'].sum()
        
        a_items = product_spend[product_spend['Oran'] <= 0.8]
        
        st.info(f"""
        **Dikkat:**
        Toplam maliyetinizin **%80'ini**, ürünlerinizin sadece **%{len(a_items)/len(product_spend)*100:.1f}**'i oluşturuyor.
        
        **Aksiyon:**
        Bu **{len(a_items)}** adet kritik üründe tedarikçinizle yapacağınız en ufak bir indirim pazarlığı, toplam karlılığınızı doğrudan etkiler.
        """)
        
        fig_pie = px.pie(product_spend.head(10), values='ToplamMaliyet', names='UrunKodu', title="En Çok Maliyet Yaratan Top 10 Ürün")
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: MALİYET TRENDLERİ (PPV) ---
with tab2:
    st.header("Maliyet Değişim Trendleri (Purchase Price Variance)")
    st.markdown("Tedarikçiniz zam yapıyor mu? Ürünlerin maliyeti zaman içinde nasıl değişiyor?")
    
    # Ürün Seçimi
    top_products = df.groupby('UrunKodu')['ToplamMaliyet'].sum().nlargest(50).index
    secilen_urun_ppv = st.selectbox("Maliyet trendini incelemek için ürün seçin:", top_products)
    
    if secilen_urun_ppv:
        urun_df = df[df['UrunKodu'] == secilen_urun_ppv].copy()
        urun_df['Ay'] = urun_df['Tarih'].dt.to_period('M').astype(str)
        
        # Aylık Ortalama Maliyet Hesapla
        monthly_cost = urun_df.groupby('Ay')['Maliyet'].mean().reset_index()
        
        if len(monthly_cost) > 1:
            # Trend Grafiği
            fig_line = px.line(monthly_cost, x='Ay', y='Maliyet', markers=True, title=f"'{secilen_urun_ppv}' Birim Maliyet Değişimi")
            
            # Trend Analizi (Artış/Azalış)
            ilk_fiyat = monthly_cost['Maliyet'].iloc[0]
            son_fiyat = monthly_cost['Maliyet'].iloc[-1]
            degisim = ((son_fiyat - ilk_fiyat) / ilk_fiyat) * 100
            
            if degisim > 0:
                fig_line.add_annotation(x=monthly_cost['Ay'].iloc[-1], y=son_fiyat, text=f"+%{degisim:.1f} Artış", showarrow=True, arrowhead=1)
                st.error(f"⚠️ Bu ürünün maliyeti dönem başından beri **%{degisim:.1f} artmış**. Tedarikçi ile görüşülmeli.")
            else:
                st.success(f"✅ Bu ürünün maliyeti dönem başından beri **%{abs(degisim):.1f} azalmış** veya stabil.")
                
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Bu ürün için trend oluşturacak kadar uzun vadeli veri bulunamadı.")

# --- TAB 3: KARLILIK MATRİSİ (YENİ) ---
with tab3:
    st.header("Ürün Karlılık Matrisi (Maliyet vs Kar Marjı)")
    st.markdown("Hangi ürünler hem maliyetli hem de düşük karlı? (Sorunlu Ürünler)")
    
    # Ürün bazında özet
    product_summary = df.groupby('UrunKodu').agg(
        OrtalamaMaliyet=('Maliyet', 'mean'),
        OrtalamaKarMarji=('KarMarji', 'mean'),
        ToplamHacim=('ToplamTutar', 'sum')
    ).reset_index()
    
    # Scatter Plot
    # X ekseni: Maliyet, Y ekseni: Kar Marjı, Boyut: Satış Hacmi
    fig_matrix = px.scatter(
        product_summary,
        x="OrtalamaMaliyet",
        y="OrtalamaKarMarji",
        size="ToplamHacim",
        hover_name="UrunKodu",
        title="Maliyet ve Karlılık Konumlandırması",
        labels={"OrtalamaMaliyet": "Birim Maliyet (€)", "OrtalamaKarMarji": "Kar Marjı (%)"},
        color="OrtalamaKarMarji",
        color_continuous_scale="RdYlGn"
    )
    
    # Ortalama çizgileri
    avg_margin = product_summary['OrtalamaKarMarji'].mean()
    avg_cost = product_summary['OrtalamaMaliyet'].mean()
    
    fig_matrix.add_vline(x=avg_cost, line_dash="dash", line_color="grey", annotation_text="Ort. Maliyet")
    fig_matrix.add_hline(y=avg_margin, line_dash="dash", line_color="grey", annotation_text="Ort. Marj")
    
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.info("""
    **Grafik Nasıl Yorumlanır?**
    * **Sağ Alt Köşe (Kırmızı Alan):** Yüksek Maliyetli ama Düşük Karlı ürünler. Şirket için en büyük risktir. Tedarikçiyle maliyet konuşulmalı veya satış fiyatı artırılmalı.
    * **Sol Üst Köşe (Yeşil Alan):** Düşük Maliyetli ve Yüksek Karlı ürünler. Nakit inekleridir (Cash Cows).
    """)