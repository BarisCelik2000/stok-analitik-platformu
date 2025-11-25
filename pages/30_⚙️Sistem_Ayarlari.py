# pages/30_⚙️Sistem_Ayarlari.py
from auth_manager import yetki_kontrol
from navigation import make_sidebar
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sistem Ayarları", layout="wide")
make_sidebar()
yetki_kontrol("Sistem Ayarları")

st.title("⚙️ Sistem ve Görünüm Ayarları")
st.markdown("Uygulamanın temasını, grafik renklerini ve performans ayarlarını buradan yönetebilirsiniz.")

tab1, tab2 = st.tabs(["🎨 Görünüm ve Tema", "🚀 Performans ve Önbellek"])

# --- TAB 1: GÖRÜNÜM ---
with tab1:
    st.header("Görsel Tercihler")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌓 Aydınlık / Karanlık Mod")
        st.info("""
        Streamlit, sistem temanızı otomatik algılar. Ancak manuel değiştirmek isterseniz:
        
        1. Sağ üst köşedeki **"..." (Üç Nokta)** menüsüne tıklayın.
        2. **Settings** seçeneğine gidin.
        3. **Theme** kısmından "Light" veya "Dark" seçin.
        """)
        
        

    with col2:
        st.subheader("📊 Grafik Renk Paleti")
        st.markdown("Grafiklerde kullanılan varsayılan renk setini buradan değiştirebilirsiniz.")
        
        secilen_tema = st.selectbox(
            "Grafik Teması Seçin:",
            ["Standart (Plotly)", "Kurumsal (Mavi/Gri)", "Canlı (Pastel)", "Kontrast (Siyah/Sarı)"],
            index=0
        )
        
        if st.button("Temayı Uygula"):
            st.session_state['grafik_temasi'] = secilen_tema
            st.success(f"✅ Grafik teması **'{secilen_tema}'** olarak ayarlandı. (Grafiklerin güncellenmesi için sayfayı yenileyin)")
            
        # Önizleme
        import plotly.express as px
        df_sample = pd.DataFrame({'Kategori': ['A','B','C'], 'Değer': [30, 50, 20]})
        
        template = "plotly"
        if secilen_tema == "Kurumsal (Mavi/Gri)": template = "simple_white"
        elif secilen_tema == "Canlı (Pastel)": template = "ggplot2"
        elif secilen_tema == "Kontrast (Siyah/Sarı)": template = "plotly_dark"
        
        fig = px.bar(df_sample, x='Kategori', y='Değer', title="Tema Önizleme", template=template)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PERFORMANS ---
with tab2:
    st.header("Sistem Performansı")
    
    st.warning("""
    **Önbellek (Cache) Nedir?**
    Uygulama, büyük verileri her seferinde tekrar yüklememek için hafızada tutar. 
    Eğer yeni veri eklediyseniz ve grafiklerde görünmüyorsa önbelleği temizleyin.
    """)
    
    if st.button("🧹 Önbelleği Temizle (Clear Cache)", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Tüm önbellek temizlendi! Veriler kaynaktan yeniden yüklenecek.")
        st.balloons()

    st.markdown("---")
    st.subheader("📁 Sistem Bilgisi")
    import sys
    st.json({
        "Python Versiyonu": sys.version.split()[0],
        "Streamlit Durumu": "Aktif",
        "Kullanılan Veri Kaynağı": "satis_verileri_guncellenmis.json"
    })