# pages/26_🏷️Dinamik_Fiyatlandirma.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_handler import veriyi_yukle_ve_temizle
from auth_manager import yetki_kontrol
from navigation import make_sidebar
st.set_page_config(page_title="Dinamik Fiyatlandırma", layout="wide")
make_sidebar()
yetki_kontrol("Dinamik Fiyatlandırma")

@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except:
    st.error("Veri yüklenemedi.")
    st.stop()

st.title("🏷️ Dinamik Fiyatlandırma Motoru")
st.markdown("""
Maliyet, stok baskısı, talep trendi ve rakip fiyatlarına göre, 
ürününüz için **karı veya nakit akışını maksimize eden en uygun fiyatı** önerir.
""")

# --- ÜRÜN SEÇİMİ ---
col1, col2 = st.columns([1, 2])
with col1:
    # En çok işlem gören ürünleri listele (Popüler ürünler)
    top_products = df.groupby('UrunKodu')['Miktar'].sum().nlargest(200).index
    secilen_urun = st.selectbox("Fiyatlandırılacak Ürünü Seçin:", top_products)
    
    # Ürün Verileri
    urun_df = df[df['UrunKodu'] == secilen_urun]
    mevcut_fiyat = urun_df['BirimFiyat'].mean()
    
    # Maliyet (Veride yoksa %75 varsayımı)
    if 'Maliyet' in df.columns:
        maliyet = urun_df['Maliyet'].mean()
    else:
        maliyet = mevcut_fiyat * 0.75

with col2:
    kar_marji_mevcut = ((mevcut_fiyat - maliyet) / mevcut_fiyat) * 100
    st.info(f"""
    **Ürün Künyesi:**
    * 📦 **Mevcut Satış Fiyatı:** {mevcut_fiyat:.2f} €
    * 🏭 **Birim Maliyet:** {maliyet:.2f} €
    * 📊 **Mevcut Kar Marjı:** %{kar_marji_mevcut:.1f}
    """)

st.markdown("---")

# --- SİMÜLASYON PARAMETRELERİ (DIŞ ETKENLER) ---
st.subheader("Piyasa Koşulları ve Strateji")
c1, c2, c3 = st.columns(3)

# 1. Rakip Fiyatı (Kullanıcı Girdisi)
rakip_fiyati = c1.number_input(
    "En Güçlü Rakip Fiyatı (€)", 
    value=float(mevcut_fiyat), 
    step=0.5,
    help="Rakipleriniz bu ürünü kaça satıyor?"
)

# 2. Stok Durumu (Stok Baskısı)
# Stok çoksa fiyatı düşür (Erit), azsa artır (Kar Et)
stok_durumu = c2.select_slider(
    "Mevcut Stok Seviyesi",
    options=["Kritik (Çok Az)", "Düşük", "Normal", "Yüksek", "Aşırı (Stok Şişkin)"],
    value="Normal"
)

# 3. Talep Trendi
talep_trendi = c3.select_slider(
    "Piyasa Talep Trendi",
    options=["Çok Düşük (Ölü Sezon)", "Düşük", "Normal", "Yüksek", "Patlama (Sezon)"],
    value="Normal"
)

# --- FİYATLANDIRMA ALGORİTMASI ---
def fiyat_onerisi_hesapla(baz_fiyat, maliyet, rakip, stok, talep):
    # Başlangıç: Rekabetçi olmak için rakip fiyatı baz alalım
    # (Strateji: Rakipten %2 ucuz olmaya çalış, ama faktörlere göre değiş)
    onerilen_fiyat = rakip
    
    # 1. Stok Etkisi (Stok maliyetini yönetmek için)
    stok_katsayilari = {
        "Kritik (Çok Az)": 1.15,  # Stok azsa fiyatı artır (Kıtlık İlkesi)
        "Düşük": 1.05,
        "Normal": 1.00,
        "Yüksek": 0.95,           # Stok fazlaysa indirim yap
        "Aşırı (Stok Şişkin)": 0.85 # Acil elden çıkar
    }
    onerilen_fiyat *= stok_katsayilari[stok]
    
    # 2. Talep Etkisi (Talebe göre esneklik)
    talep_katsayilari = {
        "Çok Düşük (Ölü Sezon)": 0.90,
        "Düşük": 0.95,
        "Normal": 1.00,
        "Yüksek": 1.05,
        "Patlama (Sezon)": 1.20 # Talep patlıyorsa karı maksimize et
    }
    onerilen_fiyat *= talep_katsayilari[talep]
    
    # 3. Güvenlik Sınırı (Asla zararına satma - En az %5 kar bırak)
    min_guvenli_fiyat = maliyet * 1.05
    
    # Eğer önerilen fiyat maliyetin altına düşerse, taban fiyata çek
    if onerilen_fiyat < min_guvenli_fiyat:
        onerilen_fiyat = min_guvenli_fiyat
        
    return onerilen_fiyat

# Hesaplama
yeni_fiyat = fiyat_onerisi_hesapla(mevcut_fiyat, maliyet, rakip_fiyati, stok_durumu, talep_trendi)
yeni_marj = ((yeni_fiyat - maliyet) / yeni_fiyat) * 100

# --- SONUÇ GÖSTERİMİ ---
st.markdown("---")
st.subheader("🎯 Yapay Zeka Fiyat Önerisi")

col_res1, col_res2, col_res3 = st.columns(3)

# Değişim oranı
degisim = ((yeni_fiyat - mevcut_fiyat) / mevcut_fiyat) * 100
renk = "off"
if degisim > 0: renk = "normal"   # Fiyat artışı (Yeşil)
elif degisim < 0: renk = "inverse" # Fiyat düşüşü (Kırmızı)

col_res1.metric("Önerilen Satış Fiyatı", f"{yeni_fiyat:.2f} €", f"%{degisim:.1f}", delta_color=renk)
col_res2.metric("Tahmini Yeni Kar Marjı", f"%{yeni_marj:.1f}")

fark_rakip = yeni_fiyat - rakip_fiyati
durum_rakip = "Rakipten Pahalı" if fark_rakip > 0 else "Rakipten Ucuz"
col_res3.metric("Rekabet Durumu", durum_rakip, f"{fark_rakip:.2f} € Fark")

# --- GÖRSEL KARŞILAŞTIRMA ---
fig = go.Figure()

x_labels = ['Maliyet', 'Mevcut Fiyat', 'Rakip Fiyatı', 'Önerilen Fiyat']
y_values = [maliyet, mevcut_fiyat, rakip_fiyati, yeni_fiyat]
colors = ['gray', 'blue', 'orange', 'green']

fig.add_trace(go.Bar(
    x=x_labels,
    y=y_values,
    marker_color=colors,
    text=[f"{v:.2f}€" for v in y_values],
    textposition='auto'
))

fig.update_layout(title="Fiyatlandırma Stratejisi Karşılaştırması", yaxis_title="Fiyat (€)")
st.plotly_chart(fig, use_container_width=True)

# --- STRATEJİK AÇIKLAMA ---
st.success(f"""
### 💡 Neden Bu Fiyat?
Yapay zeka algoritması şu kararları verdi:
1.  **Stok Etkisi:** Stok durumunuz **'{stok_durumu}'** olduğu için fiyatta {'artış' if stok_durumu in ['Kritik (Çok Az)', 'Düşük'] else 'indirim'} yönlü baskı oluştu.
2.  **Talep Etkisi:** Piyasa talebi **'{talep_trendi}'** seviyesinde olduğu için {'ekstra kar marjı eklendi' if talep_trendi in ['Yüksek', 'Patlama (Sezon)'] else 'fiyat rekabetçi tutuldu'}.
3.  **Güvenlik:** Fiyatın maliyetiniz olan **{maliyet:.2f} €** seviyesinin altına düşmesi engellendi.
""")