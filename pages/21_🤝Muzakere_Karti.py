# pages/22_🤝Muzakere_Karti.py

import streamlit as st
import pandas as pd
import plotly.express as px
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
try:
    from help_content import yardim_goster
    yardim_goster("Müzakere Kartı")
except:
    pass

st.set_page_config(page_title="Müzakere Kartı", layout="wide")
make_sidebar()
yetki_kontrol("Müzakere Kartı")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

# Veri Hazırlığı
if 'Maliyet' not in df.columns:
    df['Maliyet'] = df['BirimFiyat'] * 0.75
    df['ToplamMaliyet'] = df['ToplamTutar'] * 0.75
else:
    df['ToplamMaliyet'] = df['Miktar'] * df['Maliyet']

st.title("🤝 Müzakere Hazırlık Kartı")
st.markdown("Tedarikçi görüşmesi öncesi, seçilen ürünle ilgili tüm kritik verileri tek sayfada özetler.")

# --- ÜRÜN SEÇİMİ ---
col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    # En çok maliyet yaratan ürünleri listele
    top_products = df.groupby('UrunKodu')['ToplamMaliyet'].sum().nlargest(500).index
    secilen_urun = st.selectbox("Görüşme yapılacak ürünü seçin:", top_products)

if secilen_urun:
    urun_df = df[df['UrunKodu'] == secilen_urun].copy()
    urun_df = urun_df.sort_values('Tarih')
    
    # --- TEMEL İSTATİSTİKLER ---
    toplam_alim_adedi = urun_df['Miktar'].sum()
    toplam_odenen = urun_df['ToplamMaliyet'].sum()
    ilk_alim_tarihi = urun_df['Tarih'].min()
    son_alim_tarihi = urun_df['Tarih'].max()
    
    # Fiyat Trendi
    ilk_fiyat = urun_df['Maliyet'].iloc[0]
    son_fiyat = urun_df['Maliyet'].iloc[-1]
    fiyat_degisimi = ((son_fiyat - ilk_fiyat) / ilk_fiyat) * 100
    
    # Ortalama Fiyat
    ort_fiyat = urun_df['Maliyet'].mean()
    
    st.markdown("### 1. Hacim ve Fiyat Geçmişi")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("Toplam Alım Hacmi", f"{toplam_alim_adedi:,.0f} Adet")
    kpi2.metric("Toplam Ödenen Para", f"{toplam_odenen:,.0f} €")
    kpi3.metric("Son Birim Maliyet", f"{son_fiyat:,.2f} €")
    
    delta_color = "inverse" if fiyat_degisimi > 0 else "normal" # Fiyat arttıysa kırmızı (kötü), düştüyse yeşil (iyi)
    kpi4.metric("Tarihsel Fiyat Değişimi", f"%{fiyat_degisimi:.1f}", delta_color=delta_color)

    # --- MÜZAKERE KOZLARI (LEVERAGE POINTS) ---
    st.markdown("### 2. Müzakere Kozları (Leverage Points)")
    
    kozlar = []
    
    # Koz 1: Hacim Gücü
    if toplam_alim_adedi > 1000:
        kozlar.append(f"✅ **Yüksek Hacim:** Bugüne kadar **{toplam_alim_adedi:,.0f}** adet alım yaptık. Sadık ve büyük bir müşteriyiz.")
    
    # Koz 2: Fiyat Artışı
    if fiyat_degisimi > 10:
        kozlar.append(f"⚠️ **Fiyat Artışı:** Başlangıca göre maliyetimiz **%{fiyat_degisimi:.1f}** artmış. İndirim veya sabitleme talep etmeliyiz.")
    elif fiyat_degisimi < 0:
        kozlar.append(f"👍 **Fiyat Avantajı:** Fiyatlar düşüş trendinde. Bu trendi korumalıyız.")
        
    # Koz 3: Volatilite
    std_dev = urun_df['Maliyet'].std()
    if (std_dev / ort_fiyat) > 0.15:
        kozlar.append(f"📉 **Fiyat İstikrarsızlığı:** Fiyatlar çok dalgalı. Uzun vadeli sabit fiyat anlaşması önerilebilir.")
        
    # Koz 4: Son Alım Zamanı
    gun_farki = (pd.to_datetime("today") - son_alim_tarihi).days
    if gun_farki > 90:
        kozlar.append(f"📦 **Yeniden Sipariş:** {gun_farki} gündür alım yapmadık. Yeni sipariş vereceğiz, bunu pazarlık kozu yapalım.")

    # Kozları Ekrana Bas
    for koz in kozlar:
        st.info(koz)
        
    if not kozlar:
        st.write("Belirgin bir müzakere kozu tespit edilemedi. Standart süreç işleyebilir.")

    # --- GRAFİKSEL ANALİZ ---
    st.markdown("### 3. Görsel Kanıtlar")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Fiyat Trendi
        fig_trend = px.line(urun_df, x='Tarih', y='Maliyet', title="Zaman İçinde Birim Maliyet Değişimi", markers=True)
        # Trend çizgisi ekleyelim (Kırmızı)
        fig_trend.add_hline(y=ort_fiyat, line_dash="dash", line_color="gray", annotation_text="Ortalama Maliyet")
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_g2:
        # Aylık Alım Hacmi (Seasonality)
        urun_df['Ay'] = urun_df['Tarih'].dt.month_name()
        seasonality = urun_df.groupby('Ay')['Miktar'].sum().reindex([
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).reset_index()
        
        fig_bar = px.bar(seasonality, x='Ay', y='Miktar', title="Hangi Aylarda Daha Çok Alıyoruz?", color='Miktar')
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- YAZDIRILABİLİR ÖZET ---
    st.markdown("---")
    with st.expander("🖨️ Yazdırılabilir Özet Tablo"):
        st.dataframe(urun_df[['Tarih', 'Miktar', 'BirimFiyat', 'Maliyet', 'ToplamMaliyet']].sort_values('Tarih', ascending=False).style.format({
            'BirimFiyat': '{:.2f} €', 'Maliyet': '{:.2f} €', 'ToplamMaliyet': '{:,.0f} €'
        }))