# pages/15_💰Fiyat_Esnekligi.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from auth_manager import yetki_kontrol
from data_handler import veriyi_yukle_ve_temizle
from navigation import make_sidebar
st.set_page_config(page_title="Fiyat Esnekliği Analizi", layout="wide")
make_sidebar()
yetki_kontrol("Fiyat Esnekliği Analizi")

@st.cache_data
def veriyi_getir():
    # data_handler artık tek değer (df) döndürüyor, hata almayacaksınız.
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except Exception as e:
    st.error(f"Veri hatası: {e}")
    st.stop()

st.title("💰 Fiyat Esnekliği (Price Elasticity) Analizi")
st.markdown("""
Bu modül, ürünlerinizin **fiyat değişimlerine karşı ne kadar hassas olduğunu** ölçer.
* **İnelastik (Katı) Talep:** Fiyat artsa bile satış adedi çok düşmez. (Fırsat Ürünü 💎 - Kar marjını artırabilirsiniz)
* **Elastik (Hassas) Talep:** Fiyat arttığında satış adedi sert düşer. (Dikkatli Olunmalı ⚠️ - Müşteri fiyata duyarlı)
""")

# --- VERİ HAZIRLIĞI ---
# Analiz için yeterli veri noktasına sahip ürünleri bulalım
# (En az 10 işlem görmüş ürünler)
islem_sayilari = df['UrunKodu'].value_counts()
yeterli_veri_urunler = islem_sayilari[islem_sayilari > 10].index

if len(yeterli_veri_urunler) == 0:
    st.warning("Esneklik analizi için ürünlerin yeterli tarihsel derinliği (işlem sayısı) bulunamadı.")
    st.stop()

# --- ARAYÜZ ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Ürün Seçimi")
    # Listeyi en çok işlem görene göre sıralayalım (Popüler ürünler en üstte)
    secilen_urun = st.selectbox(
        "Analiz edilecek ürünü seçin:", 
        options=yeterli_veri_urunler[:500], # Performans için ilk 500
        help="Sadece en az 10 işlemi olan ürünler listelenir."
    )

# --- ANALİZ MOTORU ---
if secilen_urun:
    # 1. Ürün verisini çek
    urun_df = df[df['UrunKodu'] == secilen_urun].copy()
    
    # Aykırı değer temizliği (Z-Score) - Aşırı uç fiyatları atalım ki analiz bozulmasın
    # (Sadece varyasyon varsa çalışır)
    if urun_df['BirimFiyat'].std() > 0:
        urun_df = urun_df[(np.abs(stats.zscore(urun_df['BirimFiyat'])) < 3)]
    
    # 2. Zaman bazlı gruplama (Aylık Ortalama Fiyat ve Toplam Miktar)
    # Günlük gürültüyü azaltmak için veriyi aylık periyotlara sıkıştırıyoruz.
    urun_df['Ay'] = urun_df['Tarih'].dt.to_period('M').astype(str)
    
    analiz_df = urun_df.groupby('Ay').agg(
        OrtalamaFiyat=('BirimFiyat', 'mean'),
        ToplamMiktar=('Miktar', 'sum')
    ).reset_index()

    # Logaritmik dönüşüm (Ekonomi literatüründe esneklik: ln(Q) = a + b * ln(P))
    # Buradaki 'b' katsayısı esnekliği verir.
    analiz_df['LogFiyat'] = np.log(analiz_df['OrtalamaFiyat'])
    analiz_df['LogMiktar'] = np.log(analiz_df['ToplamMiktar'])

    # Yeterli fiyat varyasyonu (değişkenliği) var mı?
    fiyat_varyasyonu = analiz_df['OrtalamaFiyat'].std()
    
    with col2:
        # Eğer veri noktası çok azsa veya fiyat hiç değişmemişse analiz yapılamaz
        if fiyat_varyasyonu < 0.01 or len(analiz_df) < 3:
            st.info(f"⚠️ **{secilen_urun}** için yeterli fiyat değişimi gözlemlenmedi. Esneklik hesaplanamıyor.")
            st.caption("Bir ürünün fiyat esnekliğini ölçmek için, geçmişte farklı fiyatlardan satılmış olması ve en az 3 farklı dönem verisi gerekir.")
            
            # Yine de satış grafiğini gösterelim
            fig_basic = px.line(analiz_df, x='Ay', y=['OrtalamaFiyat', 'ToplamMiktar'], markers=True, 
                                title="Fiyat ve Miktar Değişimi (Yeterli Varyasyon Yok)")
            st.plotly_chart(fig_basic, use_container_width=True)
            
        else:
            # 3. Regresyon Hesaplama (Slope = Esneklik)
            slope, intercept, r_value, p_value, std_err = stats.linregress(analiz_df['LogFiyat'], analiz_df['LogMiktar'])
            esneklik = slope
            r_kare = r_value**2

            # --- SONUÇ KARTLARI ---
            st.subheader("Analiz Sonuçları")
            kpi1, kpi2, kpi3 = st.columns(3)
            
            kpi1.metric("Fiyat Esneklik Katsayısı", f"{esneklik:.2f}")
            kpi2.metric("Model Güvenilirliği (R²)", f"{r_kare:.2f}", help="1'e ne kadar yakınsa, fiyat-miktar ilişkisi o kadar güçlüdür.")
            
            # Yorumlama Mantığı
            if esneklik > -1:
                durum = "İNELASTİK (Katı) Talep 💎"
                aciklama = "Müşteri fiyata çok duyarlı değil. Fiyatı artırmak toplam karı artırabilir."
                renk = "green"
            elif esneklik < -1:
                durum = "ELASTİK (Hassas) Talep ⚠️"
                aciklama = "Müşteri fiyata karşı çok hassas. Küçük bir zam, satış adedini ciddi oranda düşürebilir."
                renk = "red"
            else:
                durum = "BİRİM Esneklik ⚖️"
                aciklama = "Fiyat değişimi, satış adedini aynı oranda ters etkiliyor."
                renk = "orange"
                
            kpi3.markdown(f":{renk}[**{durum}**]")
            st.success(f"💡 **Yorum:** {aciklama}")

            # --- GRAFİKLER ---
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("**Talep Eğrisi (Fiyat vs Miktar)**")
                # Scatter plot + Trendline
                fig_scatter = px.scatter(
                    analiz_df, x='OrtalamaFiyat', y='ToplamMiktar', 
                    trendline="ols", # Otomatik regresyon çizgisi
                    hover_data=['Ay'],
                    title=f"Talep Eğrisi (Eğim: {esneklik:.2f})",
                    labels={'OrtalamaFiyat': 'Fiyat (€)', 'ToplamMiktar': 'Satış Adedi'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col_g2:
                st.markdown("**Zaman İçinde Fiyat ve Miktar İlişkisi**")
                # İki eksenli grafik
                fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Satış Adedi (Sol Eksen)
                fig_dual.add_trace(
                    go.Scatter(x=analiz_df['Ay'], y=analiz_df['ToplamMiktar'], name="Satış Adedi", mode='lines+markers', line=dict(color='#636EFA')),
                    secondary_y=False
                )
                # Fiyat (Sağ Eksen)
                fig_dual.add_trace(
                    go.Scatter(x=analiz_df['Ay'], y=analiz_df['OrtalamaFiyat'], name="Fiyat", mode='lines+markers', line=dict(color='#00CC96', dash='dot')),
                    secondary_y=True
                )
                
                fig_dual.update_layout(title="Zaman İçinde Fiyat ve Talep Değişimi")
                fig_dual.update_yaxes(title_text="Satış Adedi", secondary_y=False)
                fig_dual.update_yaxes(title_text="Fiyat (€)", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_dual, use_container_width=True)

            # --- SİMÜLATÖR ---
            st.markdown("---")
            st.subheader("🧮 Fiyat Değişimi Simülatörü")
            st.markdown("Eğer bu ürünün fiyatını değiştirirseniz tahmini senaryo ne olur?")
            
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                fiyat_degisim_yuzdesi = st.slider("Fiyatı Yüzde Kaç Değiştireceksiniz?", -50, 50, 10, step=5)
            
            with col_sim2:
                # Elastikiyet Formülü: %Q = Elastikiyet * %P
                beklenen_miktar_degisimi = esneklik * fiyat_degisim_yuzdesi
                
                # Ciro Etkisi: Yeni Ciro = (1 + %P) * (1 + %Q) * Eski Ciro
                # Matematiksel olarak ciro etkisi bu formülle hesaplanır.
                ciro_etkisi = ((1 + fiyat_degisim_yuzdesi/100) * (1 + beklenen_miktar_degisimi/100)) - 1
                
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Beklenen Satış Adedi Değişimi", f"%{beklenen_miktar_degisimi:.1f}", delta_color="normal")
                col_res2.metric("Beklenen Ciro Etkisi", f"%{ciro_etkisi*100:.1f}", 
                         delta_color="normal" if ciro_etkisi > 0 else "inverse")
                
                if ciro_etkisi > 0:
                    st.success("✅ Bu fiyat değişikliği **Toplam Cironuzu Artırabilir**.")
                else:
                    st.error("📉 Bu fiyat değişikliği **Ciro Kaybına Neden Olabilir**.")