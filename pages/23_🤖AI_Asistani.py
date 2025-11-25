# pages/23_🤖AI_Asistani.py

import streamlit as st
import pandas as pd
from data_handler import veriyi_yukle_ve_temizle
from pandasai import SmartDataframe
from auth_manager import yetki_kontrol
from gemini_adapter import GeminiAdapter 
from navigation import make_sidebar

st.set_page_config(page_title="AI Veri Asistanı", layout="wide")
make_sidebar()
yetki_kontrol("AI Veri Asistanı")

st.title("🤖 AI Veri Asistanı (Google Gemini)")

# --- VERİ YÜKLEME ---
@st.cache_data
def veriyi_getir():
    return veriyi_yukle_ve_temizle('satis_verileri_guncellenmis.json')

try:
    df = veriyi_getir()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

# --- API KEY ---
st.markdown("---")
api_key = st.sidebar.text_input("Google AI Studio Key", type="password")

if not api_key:
    st.info("Lütfen Google API anahtarınızı girin.")
    st.stop()

# --- SOHBET MOTORU ---
try:
    # Kendi yazdığımız sağlam adaptörü kullanıyoruz
    llm = GeminiAdapter(api_key=api_key)
    
    smart_df = SmartDataframe(df, config={
        "llm": llm,
        "verbose": True,
        "open_charts": False
    })
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area("Sorunuzu buraya yazın:", height=100, placeholder="Örn: En çok kar eden 5 ürünü grafik olarak göster.")
        
        if st.button("Analiz Et 🚀", type="primary"):
            if user_query:
                with st.spinner("Yapay zeka veriyi inceliyor..."):
                    try:
                        response = smart_df.chat(user_query)
                        
                        if response is not None:
                            st.success("İşlem Başarılı!")
                            
                            # Cevap Türüne Göre Gösterim
                            if isinstance(response, str) and ("png" in response or "jpg" in response):
                                st.image(response)
                            elif isinstance(response, (pd.DataFrame, pd.Series)):
                                st.dataframe(response)
                            else:
                                st.write(response)
                            
                            with st.expander("Python Kodu"):
                                st.code(smart_df.last_code_generated)
                        else:
                            st.warning("Cevap üretilemedi.")
                            
                    except Exception as e:
                        st.error(f"Hata: {e}")
            else:
                st.warning("Soru girmediniz.")

    with col2:
        st.info("**İpucu:** 'Bana aylık satış grafiğini çiz' gibi Türkçe komutlar verebilirsiniz.")

except Exception as e:
    st.error(f"Model başlatma hatası: {e}")