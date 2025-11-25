# pages/17_🎨Gelismis_Gorseller.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from analysis_engine import rfm_skorlarini_hesapla, musterileri_segmentle
from navigation import make_sidebar
st.set_page_config(page_title="Gelişmiş Görseller", layout="wide")
make_sidebar()
yetki_kontrol("Gelişmiş Görseller")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
    rfm_df = rfm_skorlarini_hesapla(df)
    sonuclar_df = musterileri_segmentle(rfm_df)
except Exception as e:
    st.error(f"Veri hatası: {e}")
    st.stop()

st.title("🎨 Gelişmiş Görselleştirme ve Mikro Analizler")
st.markdown("Veri setindeki gizli desenleri, yoğunlukları ve dağılımları keşfedin.")

tab1, tab2, tab3 = st.tabs(["🌌 3D Müşteri Uzayı", "🔥 Zaman Isı Haritası", "⚖️ Müşteri Konsantrasyonu"])

# --- TAB 1: 3D RFM ANALİZİ ---
with tab1:
    st.header("3D RFM Müşteri Uzayı")
    st.markdown("Müşteri segmentlerinin Recency, Frequency ve Monetary eksenlerinde nasıl kümelendiğini inceleyin.")
    
    # Performans için örnekleme yapalım (Çok fazla nokta tarayıcıyı yorar)
    if len(sonuclar_df) > 2000:
        plot_df = sonuclar_df.sample(2000, random_state=42)
        st.caption("ℹ️ Performans için rastgele 2000 müşteri gösterilmektedir.")
    else:
        plot_df = sonuclar_df

    fig_3d = px.scatter_3d(
        plot_df, 
        x='Recency', 
        y='Frequency', 
        z='Monetary',
        color='Segment',
        opacity=0.7,
        size_max=10,
        hover_name=plot_df.index,
        title="3D Segment Dağılımı",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=700)
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 2: ZAMAN ISI HARİTASI ---
with tab2:
    st.header("Satışların Zamansal Yoğunluğu")
    
    # Veride saat bilgisi var mı kontrol et
    df['Saat'] = df['Tarih'].dt.hour
    df['Gun'] = df['Tarih'].dt.day_name()
    
    # Türkçe gün isimleri için map
    gun_map = {
        'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba', 
        'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi', 'Sunday': 'Pazar'
    }
    df['Gun_Tr'] = df['Gun'].map(gun_map)
    gun_sirasi = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']

    # Eğer tüm saatler 0 ise (Veri setinde saat yoksa), Ay vs Gün analizi yap
    if df['Saat'].sum() == 0:
        st.info("ℹ️ Veri setinde saat detayı bulunamadı. Analiz **Gün vs Ay** bazında yapılıyor.")
        df['Ay'] = df['Tarih'].dt.month_name()
        heatmap_data = df.groupby(['Gun_Tr', 'Ay']).size().reset_index(name='IslemSayisi')
        x_ekseni = 'Ay'
        baslik = "Ay ve Gün Bazlı Satış Yoğunluğu"
    else:
        st.info("ℹ️ Veri setinde saat detayı mevcut. Analiz **Saat vs Gün** bazında yapılıyor.")
        heatmap_data = df.groupby(['Gun_Tr', 'Saat']).size().reset_index(name='IslemSayisi')
        x_ekseni = 'Saat'
        baslik = "Haftanın Günleri ve Saatlere Göre Satış Yoğunluğu"

    fig_heat = px.density_heatmap(
        heatmap_data, 
        x=x_ekseni, 
        y='Gun_Tr', 
        z='IslemSayisi', 
        nbinsx=24, 
        color_continuous_scale='Viridis',
        title=baslik,
        category_orders={'Gun_Tr': gun_sirasi}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: LORENZ EĞRİSİ ---
with tab3:
    st.header("Müşteri Gelir Konsantrasyonu (Lorenz Eğrisi)")
    st.markdown("Cironun ne kadarı, müşterilerin ne kadarı tarafından oluşturuluyor? (Gelir Adaletsizliği)")
    
    # Veriyi hazırla
    lorenz_df = sonuclar_df[['Monetary']].sort_values('Monetary').copy()
    
    # Kümülatif toplamlar
    lorenz_df['Kumulatif_Musteri_Orani'] = np.arange(1, len(lorenz_df) + 1) / len(lorenz_df)
    lorenz_df['Kumulatif_Ciro'] = lorenz_df['Monetary'].cumsum()
    lorenz_df['Kumulatif_Ciro_Orani'] = lorenz_df['Kumulatif_Ciro'] / lorenz_df['Monetary'].sum()
    
    # Eşit dağılım çizgisi (Her müşteri eşit ciro yapsaydı)
    esit_dagilim = pd.DataFrame({
        'Kumulatif_Musteri_Orani': [0, 1],
        'Kumulatif_Ciro_Orani': [0, 1],
        'Tip': 'İdeal Eşitlik'
    })
    
    fig_lorenz = go.Figure()
    
    # Gerçek Veri
    fig_lorenz.add_trace(go.Scatter(
        x=lorenz_df['Kumulatif_Musteri_Orani'],
        y=lorenz_df['Kumulatif_Ciro_Orani'],
        mode='lines',
        name='Gerçek Dağılım',
        line=dict(color='red', width=3)
    ))
    
    # İdeal Eşitlik
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Tam Eşitlik (Teorik)',
        line=dict(color='grey', dash='dash')
    ))
    
    # Gini Katsayısı (Basitleştirilmiş alan hesabı)
    # Alan A = 0.5 - Eğri altındaki alan
    # Gini = A / 0.5 = 2 * A
    alan = np.trapz(lorenz_df['Kumulatif_Ciro_Orani'], lorenz_df['Kumulatif_Musteri_Orani'])
    gini = 1 - 2 * alan
    
    fig_lorenz.update_layout(
        title=f"Lorenz Eğrisi (Gini Katsayısı: {gini:.2f})",
        xaxis_title="Müşterilerin Kümülatif %'si (En az harcayandan en çoka)",
        yaxis_title="Cironun Kümülatif %'si",
        height=600
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig_lorenz, use_container_width=True)
    with col2:
        st.info(f"""
        **Gini Katsayısı: {gini:.2f}**
        
        * **0'a yakınsa:** Ciro müşterilere eşit dağılmıştır. (Sağlıklı, tek bir müşteriye bağımlılık yok).
        * **1'e yakınsa:** Cironun neredeyse tamamını çok az sayıda müşteri yapıyordur. (Yüksek risk, o müşteriler giderse şirket batabilir).
        """)