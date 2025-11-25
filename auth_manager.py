# auth_manager.py

import streamlit as st
import time

# --- DEMO KULLANICILAR ---
# Gerçek hayatta burası bir Veritabanına bağlanır.
# Şimdilik basit bir sözlük (Dictionary) kullanıyoruz.
# Format: "kullanici_adi": "sifre"
KULLANICILAR = {
    "admin": "admin123",      # Tam Yetkili
    "baris": "1234",          # Yönetici
    "misafir": "misafir",     # Kısıtlı
    "satis": "satis2025",     # Satış Ekibi
    "satinalma": "alim2025"   # Satınalma Ekibi
}

def oturum_kontrol():
    """
    Kullanıcı giriş yapmış mı kontrol eder.
    Giriş yapmamışsa False, yapmışsa True döner.
    """
    if "giris_yapildi" not in st.session_state:
        st.session_state["giris_yapildi"] = False
        st.session_state["kullanici_adi"] = None
        
    return st.session_state["giris_yapildi"]

def giris_ekrani():
    """
    Giriş ekranını çizer ve şifre kontrolü yapar.
    """
    st.markdown("## 🔒 Güvenli Giriş Paneli")
    st.info("Lütfen devam etmek için kimliğinizi doğrulayın.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            kullanici = st.text_input("Kullanıcı Adı")
            sifre = st.text_input("Şifre", type="password")
            submit = st.form_submit_button("Giriş Yap 🚀", use_container_width=True)
            
            if submit:
                if kullanici in KULLANICILAR and KULLANICILAR[kullanici] == sifre:
                    st.session_state["giris_yapildi"] = True
                    st.session_state["kullanici_adi"] = kullanici
                    st.success(f"Hoşgeldin {kullanici}! Yönlendiriliyorsunuz...")
                    time.sleep(1)
                    st.rerun() # Sayfayı yenile
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

def cikis_yap_butonu():
    """
    Sidebar'a çıkış yap butonu ekler.
    """
    with st.sidebar:
        st.markdown("---")
        st.write(f"👤 Aktif Kullanıcı: **{st.session_state.get('kullanici_adi', 'Bilinmiyor')}**")
        if st.button("🚪 Çıkış Yap", type="primary"):
            st.session_state["giris_yapildi"] = False
            st.session_state["kullanici_adi"] = None
            st.rerun()

def yetki_kontrol(sayfa_adi):
    """
    Her sayfanın en başına konulacak bekçi fonksiyonu.
    Giriş yapılmamışsa kodu durdurur ve giriş ekranını gösterir.
    """
    # 1. Giriş Kontrolü
    if not oturum_kontrol():
        st.set_page_config(page_title="Giriş Yapın", layout="centered")
        giris_ekrani()
        st.stop() # Kodun geri kalanını çalıştırma!
    
    # 2. Giriş Yapıldıysa Çıkış Butonunu Göster
    cikis_yap_butonu()
    
    # 3. (Opsiyonel) Rol Bazlı Erişim Kontrolü
    # Örnek: 'misafir' kullanıcısı 'Maliyet Analizi' sayfasına giremesin
    user = st.session_state["kullanici_adi"]
    
    # Yasaklı Sayfa Tanımları
    yasaklar = {
        "misafir": ["Maliyet Analizi", "Nakit Akışı", "Müzakere Kartı", "EOQ Optimizasyonu"],
        "satis": ["EOQ Optimizasyonu", "Müzakere Kartı"],
        "satinalma": ["Churn Analizi", "Pazarlama ROI"]
    }
    
    # Sayfa adı yasaklı listede mi?
    if user in yasaklar:
        # Sayfa adının içinde yasaklı kelime geçiyor mu?
        for yasak_kelime in yasaklar[user]:
            if yasak_kelime in sayfa_adi:
                st.error(f"⛔ Yetkisiz Erişim: '{user}' kullanıcısı bu sayfayı görüntüleme yetkisine sahip değildir.")
                st.stop()