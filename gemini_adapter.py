# gemini_adapter.py

import google.generativeai as genai
from pandasai.llm import LLM
import re

class GeminiAdapter(LLM):
    """
    PandasAI için Google Gemini Adaptörü (Liste Tabanlı).
    - Manuel model ismi denemez.
    - Google hesabınızdaki tanımlı modelleri çeker.
    - 'generateContent' destekleyen İLK modeli kullanır.
    """
    def __init__(self, api_key, model=None):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        self.model_name = None
        self.client = None
        
        print("🔍 Google hesabındaki aktif modeller taranıyor...")
        
        try:
            # Google'dan model listesini iste
            for m in genai.list_models():
                # Sadece içerik üretimi destekleyen modellere bak
                if 'generateContent' in m.supported_generation_methods:
                    # Öncelik: Flash > Pro 1.5 > Pro 1.0 > Diğerleri
                    # Bu isimleri listede görürsek hemen kapıyoruz
                    if 'flash' in m.name:
                        self.model_name = m.name
                        break
                    elif '1.5-pro' in m.name and not self.model_name:
                        self.model_name = m.name
                    elif 'gemini-pro' in m.name and not self.model_name:
                        self.model_name = m.name
            
            # Eğer döngü bittiğinde hala model seçilmediyse, listenin ilkini al
            if not self.model_name:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        self.model_name = m.name
                        break
            
            if self.model_name:
                print(f"✅ SEÇİLEN MODEL: {self.model_name}")
                self.client = genai.GenerativeModel(self.model_name)
            else:
                raise ValueError("Hesabınızda uygun bir Gemini modeli bulunamadı.")

        except Exception as e:
            print(f"❌ Model listeleme hatası: {str(e)}")
            # Son çare fallback
            self.model_name = "models/gemini-1.5-flash"
            self.client = genai.GenerativeModel(self.model_name)

        # Güvenlik ayarları
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def call(self, instruction, value=None, suffix=""):
        prompt = str(instruction)
        if value is not None:
            prompt += f"\n{str(value)}"
        if suffix:
            prompt += f"\n{suffix}"
            
        # Net kod talimatı
        system_message = """
        You are a Python Data Analyst.
        Generate Python code to analyze the dataframe 'df'.
        Rules:
        1. Return ONLY the code. No markdown, no explanation.
        2. Use 'print()' to output text answers.
        3. Use 'st.write()', 'st.dataframe()' or 'st.pyplot()' for output if streamlit is available.
        4. If plotting, create the figure and use st.pyplot(plt.gcf()) or similar.
        """
        
        full_prompt = system_message + "\n\nQUERY:\n" + prompt
        
        try:
            response = self.client.generate_content(
                full_prompt, 
                safety_settings=self.safety_settings
            )
            
            text = response.text
            
            # Markdown temizliği
            # ```python ... ``` bloklarını temizle
            match = re.search(r"```python\s*(.*)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                text = text.replace("```", "").strip()
            
            if text.startswith("python"):
                text = text[6:].strip()
                
            return text
            
        except Exception as e:
            # Hata durumunda PandasAI'ın anlayacağı bir kod döndür
            # "No result returned" hatasını engellemek için bir print koyuyoruz
            return f'print("Üzgünüm, API şu hatayı verdi: {str(e)}")'

    @property
    def type(self) -> str:
        return "gemini-adapter"