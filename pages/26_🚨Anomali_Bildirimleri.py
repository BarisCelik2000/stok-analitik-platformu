# pages/26_🚨Anomali_Bildirimleri.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import io # Excel dönüşümü için gerekli
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
st.set_page_config(page_title="Anomali Bildirim Merkezi", layout="wide")
make_sidebar()
yetki_kontrol("Anomali Bildirim Merkezi")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

# --- EXCEL DÖNÜŞÜM FONKSİYONU ---
@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

st.title("🚨 Akıllı Bildirim Merkezi (Alerting)")
st.markdown("""
Bu modül, tüm verilerinizi tarar ve **kritik eşikleri aşan riskleri** otomatik tespit eder.
Riskli durumlar için detaylı raporları buradan indirebilirsiniz.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Alarm Eşikleri")
    esik_stok = st.slider("Kritik Stok Seviyesi (Adet)", 0, 100, 20)
    esik_satis_dusus = st.slider("Ani Satış Düşüşü (%)", 10, 50, 20)
    esik_churn_gun = st.number_input("Churn Riski (Gün)", value=90)

# --- ANALİZ MOTORU ---
alarmlar = []

# 1. STOK ALARMLARI (Ürünleri İndir)
urun_performans = df.groupby('UrunKodu')['Miktar'].sum().sort_values(ascending=False).reset_index()
kritik_urun_limiti = int(len(urun_performans) * 0.20)
if kritik_urun_limiti < 5: kritik_urun_limiti = 5
kritik_urunler_listesi = urun_performans.head(kritik_urun_limiti)['UrunKodu'].tolist()

stok_durumu = df[df['UrunKodu'].isin(kritik_urunler_listesi)].groupby('UrunKodu')['Miktar'].sum().reset_index()
stok_durumu['TahminiStok'] = stok_durumu['Miktar'] * 0.05 

riskli_urunler_df = stok_durumu[stok_durumu['TahminiStok'] <= esik_stok].sort_values('TahminiStok')

if not riskli_urunler_df.empty:
    riskli_sayi = len(riskli_urunler_df)
    alarmlar.append({
        "Tip": "KRİTİK STOK",
        "Önem": "Yüksek",
        "Mesaj": f"Toplam **{riskli_sayi}** adet 'Çok Satan' ürünün stoğu kritik seviyenin altına düştü!",
        # Excel İndirme Verileri
        "ButonEtiketi": "📥 Riskli Ürünleri İndir (Excel)",
        "DosyaAdi": "kritik_stok_listesi.xlsx",
        "Veri": riskli_urunler_df,
        "DetayTablo": riskli_urunler_df[['UrunKodu', 'TahminiStok']]
    })

# 2. SATIŞ ALARMLARI (Finansal Tablo İndir)
df['Tarih'] = pd.to_datetime(df['Tarih'])
son_tarih = df['Tarih'].max()
gecen_hafta_basi = son_tarih - timedelta(days=7)
onceki_hafta_basi = gecen_hafta_basi - timedelta(days=7)

bu_hafta_satis = df[df['Tarih'] >= gecen_hafta_basi]['ToplamTutar'].sum()
onceki_hafta_satis = df[(df['Tarih'] >= onceki_hafta_basi) & (df['Tarih'] < gecen_hafta_basi)]['ToplamTutar'].sum()

if onceki_hafta_satis > 0:
    degisim = ((bu_hafta_satis - onceki_hafta_satis) / onceki_hafta_satis) * 100
    if degisim < -esik_satis_dusus:
        # İndirilecek Finansal Tabloyu Hazırla (Son 2 haftanın verisi)
        finansal_tablo = df[(df['Tarih'] >= onceki_hafta_basi)].copy()
        finansal_tablo['Donem'] = finansal_tablo['Tarih'].apply(lambda x: 'Bu Hafta' if x >= gecen_hafta_basi else 'Geçen Hafta')
        
        alarmlar.append({
            "Tip": "SATIŞ DÜŞÜŞÜ",
            "Önem": "Orta",
            "Mesaj": f"Haftalık ciroda geçen haftaya göre **%{abs(degisim):.1f}** ani düşüş tespit edildi.",
            # Excel İndirme Verileri
            "ButonEtiketi": "📥 Finansal Tabloyu İndir (Excel)",
            "DosyaAdi": "satis_dusus_analizi.xlsx",
            "Veri": finansal_tablo[['Tarih', 'Donem', 'UrunKodu', 'Miktar', 'ToplamTutar']],
            "DetayTablo": None # Detay tablosu yoksa None geç
        })

# 3. MÜŞTERİ ALARMLARI (Müşteri Listesini İndir)
son_alimlar = df.groupby('MusteriID')['Tarih'].max().reset_index()
son_alimlar['GecenGun'] = (son_tarih - son_alimlar['Tarih']).dt.days
riskli_musteriler_df = son_alimlar[son_alimlar['GecenGun'] > esik_churn_gun]
riskli_musteri_sayisi = len(riskli_musteriler_df)

if riskli_musteri_sayisi > 0:
    alarmlar.append({
        "Tip": "MÜŞTERİ RİSKİ",
        "Önem": "Orta",
        "Mesaj": f"Toplam **{riskli_musteri_sayisi}** adet müşteri {esik_churn_gun} gündür uğramıyor.",
        # Excel İndirme Verileri
        "ButonEtiketi": "📥 Müşteri Listesini İndir (Excel)",
        "DosyaAdi": "churn_riski_musteriler.xlsx",
        "Veri": riskli_musteriler_df,
        "DetayTablo": riskli_musteriler_df[['MusteriID', 'GecenGun']].sort_values('GecenGun', ascending=False).head(20)
    })

# --- ARAYÜZ GÖSTERİMİ ---
c1, c2, c3 = st.columns(3)
df_alarmlar = pd.DataFrame(alarmlar)

if not df_alarmlar.empty:
    yuksek_risk = len(df_alarmlar[df_alarmlar['Önem'] == 'Yüksek'])
    orta_risk = len(df_alarmlar[df_alarmlar['Önem'] == 'Orta'])
    c1.metric("Aktif Bildirimler", len(df_alarmlar))
    c2.metric("🔴 Acil (Yüksek)", yuksek_risk)
    c3.metric("🟠 Uyarı (Orta)", orta_risk)
else:
    st.success("✅ Sistemde şu an aktif bir risk bulunmuyor.")

st.markdown("---")

# Alarm Listesi
if not df_alarmlar.empty:
    st.subheader("📋 Bildirim Akışı")
    
    for i, alarm in enumerate(alarmlar):
        icon = "🔴" if alarm['Önem'] == "Yüksek" else "🟠"
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 5px solid {('red' if alarm['Önem'] == 'Yüksek' else 'orange')}; 
                        background-color: #f8f9fa; 
                        padding: 15px; 
                        margin-bottom: 10px; 
                        border-radius: 5px;
                        color: #333333;"> 
                <h4 style="margin:0; color:#333333;">{icon} {alarm['Tip']}</h4>
                <p style="font-size:1.1em; margin:5px 0; color:#333333;">{alarm['Mesaj']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detay Tablo (Varsa)
            if alarm.get("DetayTablo") is not None:
                with st.expander(f"📋 Önizleme ({len(alarm['DetayTablo'])} Kayıt)"):
                    st.dataframe(alarm['DetayTablo'], use_container_width=True)
            
            # İNDİRME BUTONU
            # Veriyi Excel byte'larına çevir
            excel_data = convert_df_to_excel(alarm["Veri"])
            
            col_space, col_btn = st.columns([4, 2])
            with col_btn:
                st.download_button(
                    label=alarm["ButonEtiketi"],
                    data=excel_data,
                    file_name=alarm["DosyaAdi"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_btn_{i}",
                    use_container_width=True
                )
            

