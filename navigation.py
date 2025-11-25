# navigation.py

import streamlit as st
from time import sleep

def make_sidebar():
    """
    Bu fonksiyon her sayfada çağrılır.
    1. Varsayılan Streamlit menüsünü gizler.
    2. Sidebar'a 'Anasayfaya Dön' butonu ve Logo ekler.
    """
    # Varsayılan Sidebar Navigasyonunu Gizle (CSS ile)
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # Logo ve Başlık
        st.image("stok-logo2-Photoroom.png", use_container_width=True)
        st.markdown("<div style='text-align: center; color: white;'>STOK ANALİTİK PORTALI</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Anasayfaya Dön Butonu
        # Mevcut sayfa Anasayfa değilse göster
        # (Dosya adını kontrol edemediğimiz durumlarda her zaman gösteriyoruz, zararı yok)
        st.page_link("0_🔍Genel_Bakis.py", label="🏠 Anasayfaya Dön", icon="↩️")
        
        st.markdown("---")
        
        # Diğer Yardımcı Linkler (İsterseniz)
        st.page_link("pages/29_🎨Benim_Panom.py", label="Benim Panom", icon="🎨")
        st.page_link("pages/30_⚙️Sistem_Ayarlari.py", label="Ayarlar", icon="⚙️")