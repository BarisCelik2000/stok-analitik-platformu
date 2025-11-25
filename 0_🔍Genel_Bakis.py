# 0_🔍Genel_Bakis.py

import streamlit as st
from navigation import make_sidebar
from auth_manager import yetki_kontrol
# Sayfa Ayarı
st.set_page_config(page_title="Ana Menü", layout="wide")

# Navigasyonu Yükle (Sidebar'ı gizler)
make_sidebar()
yetki_kontrol("Ana Menü")
# Başlık
st.title("🏢 Kurumsal Analitik Portalı")
st.markdown("Lütfen işlem yapmak istediğiniz modülü seçiniz.")
st.markdown("---")

# --- MODÜL GRUPLARI ---

# 1. GRUP: SATIŞ VE MÜŞTERİ
st.subheader("🛍️ Satış ve Müşteri Yönetimi")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("📦 **Ürün Analizi**")
    st.caption("Ürün performansı, Pareto ve Birliktelik analizi.")
    st.page_link("pages/1_📦Urun_Analizi.py", label="Modüle Git", icon="👉")

with col2:
    st.info("👤 **Müşteri Detayı**")
    st.caption("Müşteri 360, RFM skorları ve geçmiş işlemler.")
    st.page_link("pages/2_👤Musteri_Detayi.py", label="Modüle Git", icon="👉")

with col3:
    st.info("📉 **Churn Analizi**")
    st.caption("Müşteri kayıp riski ve neden analizi (SHAP).")
    st.page_link("pages/6_📉Churn_Analizi.py", label="Modüle Git", icon="👉")

with col4:
    st.info("🎯 **Pazarlama ROI**")
    st.caption("Kampanya simülasyonu ve indirim optimizasyonu.")
    st.page_link("pages/7_🎯Pazarlama_ROI.py", label="Modüle Git", icon="👉")

# Alt Grup
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/3_📈Kohort_Analizi.py", label="Kohort Analizi", icon="📅")
with c2:
    st.page_link("pages/4_🗺️Musteri_Yolculugu.py", label="Müşteri Yolculuğu", icon="🗺️")
with c3:
    st.page_link("pages/8_👥Look_Alike_Analizi.py", label="Benzer Müşteri Bul", icon="👥")
with c4:
    st.page_link("pages/9_⚠️Satis_Anomalileri.py", label="Satış Anomalileri", icon="⚠️")

st.markdown("---")

# 2. GRUP: SATINALMA VE STOK
st.subheader("🏭 Satınalma ve Tedarik Zinciri")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.success("📦 **Stok ve Pareto**")
    st.caption("Stok verimliliği, ABC analizi ve Ölü Stoklar.")
    st.page_link("pages/14_📦Stok_ve_Pareto.py", label="Modüle Git", icon="👉")

with col_s2:
    st.success("💰 **Fiyat Esnekliği**")
    st.caption("Fiyat değişiminin talebe etkisi.")
    st.page_link("pages/15_💰Fiyat_Esnekligi.py", label="Modüle Git", icon="👉")

with col_s3:
    st.success("🛡️ **Güvenlik Stoğu**")
    st.caption("Ne zaman sipariş verilmeli? (ROP Hesabı).")
    st.page_link("pages/17_🛡️Guvenlik_Stogu_ROP.py", label="Modüle Git", icon="👉")

with col_s4:
    st.success("📉 **EOQ Optimizasyon**")
    st.caption("Ekonomik sipariş miktarı hesaplayıcı.")
    st.page_link("pages/18_📉EOQ_Optimizasyonu.py", label="Modüle Git", icon="👉")

# Alt Grup
cs1, cs2, cs3, cs4 = st.columns(4)
with cs1:
    st.page_link("pages/19_💸Maliyet_Analizi.py", label="Maliyet & Harcama", icon="💸")
with cs2:
    st.page_link("pages/20_💶Nakit_Akisi.py", label="Gelecek Nakit Akışı", icon="💶")
with cs3:
    st.page_link("pages/21_🤝Muzakere_Karti.py", label="Müzakere Kartı", icon="🤝")
with cs4:
    st.page_link("pages/27_📅Stok_Yaslandirma.py", label="Stok Yaşlandırma", icon="📅")

st.markdown("---")

# 3. GRUP: İLERİ ANALİTİK VE AI
st.subheader("🧠 Yapay Zeka ve Strateji")
col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    st.warning("🤖 **AI Asistanı**")
    st.caption("Verilerle doğal dilde sohbet edin.")
    st.page_link("pages/23_🤖AI_Asistani.py", label="Modüle Git", icon="👉")

with col_a2:
    st.warning("🔮 **Satış Tahmini**")
    st.caption("Gelecek dönem ciro tahminleri.")
    st.page_link("pages/5_🔮Satis_Tahminleme.py", label="Modüle Git", icon="👉")

with col_a3:
    st.warning("🏷️ **Dinamik Fiyat**")
    st.caption("AI tabanlı akıllı fiyat önerisi.")
    st.page_link("pages/25_🏷️Dinamik_Fiyatlandirma.py", label="Modüle Git", icon="👉")

with col_a4:
    st.warning("🚨 **Bildirim Merkezi**")
    st.caption("Tüm acil durum ve alarmlar.")
    st.page_link("pages/26_🚨Anomali_Bildirimleri.py", label="Modüle Git", icon="👉")

# Alt Grup
ca1, ca2, ca3, ca4 = st.columns(4)
with ca1:
    st.page_link("pages/24_🎲Monte_Carlo_Simulasyonu.py", label="Risk Simülasyonu", icon="🎲")
with ca2:
    st.page_link("pages/16_⏱️Gelecek_Satin_Alma.py", label="Zamanlama Tahmini", icon="⏱️")
with ca3:
    st.page_link("pages/10_🔬Segmentasyon_Lab.py", label="Segmentasyon Lab", icon="🔬")
with ca4:
    st.page_link("pages/13_🔀Kategori_Gecisleri.py", label="Kategori Geçişleri", icon="🔀")

st.markdown("---")

# 4. GRUP: SİSTEM VE RAPORLAMA
st.subheader("📝 Raporlama ve Sistem")
with st.expander("Tüm Rapor ve Ayarlar", expanded=True):
    c_sys1, c_sys2, c_sys3, c_sys4 = st.columns(4)
    
    with c_sys1:
        st.page_link("pages/0_📝Ozet_Rapor.py", label="Yönetici Özeti", icon="📝")
    
    with c_sys2:
        st.page_link("pages/29_🎨Benim_Panom.py", label="Benim Panom", icon="🎨")
        
    with c_sys3:
        st.page_link("pages/30_⚙️Sistem_Ayarlari.py", label="Ayarlar", icon="⚙️")
        
    with c_sys4:
        st.page_link("pages/31_🏥Veri_Sagligi.py", label="Veri Sağlığı", icon="🏥")