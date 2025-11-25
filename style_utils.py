# style_utils.py

import streamlit as st

def stili_yukle():
    st.markdown("""
    <style>
        /* Ana Arkaplan Rengi ve Font */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Metrik Kartları Tasarımı */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metrik Başlıkları */
        div[data-testid="stMetricLabel"] {
            color: #666;
            font-size: 0.9rem;
        }
        
        /* Metrik Değerleri */
        div[data-testid="stMetricValue"] {
            color: #333;
            font-weight: 700;
        }

        /* Sidebar Güzelleştirme */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        
        section[data-testid="stSidebar"] .css-17lntkn {
            color: white;
        }
        
        /* Tablo Başlıkları */
        thead tr th:first-child {display:none}
        tbody th {display:none}
        
        /* Genel Başlıklar */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #2c3e50;
        }
        
        /* Butonlar */
        .stButton button {
            border-radius: 20px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)