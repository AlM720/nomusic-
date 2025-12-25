import streamlit as st
import torch
import torchaudio
import gc
from demucs.apply import apply_model
from demucs.pretrained import get_model
import os
import subprocess
import shutil
import time
import yt_dlp

# إعداد الصفحة
st.set_page_config(page_title="عازل الموسيقى المستقر", page_icon="🎙️")

class VocalExtractor:
    def __init__(self):
        # إجبار العمل على CPU في سيرفرات Streamlit المجانية لتجنب أخطاء الـ CUDA
        self.device = torch.device("cpu")

    @st.cache_resource
    def get_model(_self):
        # استخدام htdemucs_6s مع التخزين المؤقت
        return get_model("htdemucs_6s")

    def convert_to_wav(self, input_path, output_path):
        subprocess.run(["ffmpeg", "-i", input_path, "-vn", "-ac", "2", "-ar", "44100", "-y", output_path], check=True, capture_output=True)

def download_video(url):
    output_path = "downloaded_input.mp4"
    cookies_content = st.secrets.get("coce", "")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'nocheckcertificate': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    if cookies_content:
        with open("cookies.txt", "w") as f: f.write(cookies_content)
        ydl_opts['cookiefile'] = "cookies.txt"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

st.title("🎙️ عازل الموسيقى (النسخة المصفحة)")
st.info("تم تحسين الكود ليعمل بذاكرة منخفضة لتجنب الانهيار.")

tab1, tab2 = st.tabs(["🔗 رابط", "📂 رفع ملف"])
source_path = None

with tab1:
    url_input = st.text_input("ضع الرابط هنا")
with tab2:
    uploaded_file = st.file_uploader("اختر ملف", type=["mp3", "wav", "mp4", "m4a"])

# خيار القوة (تم تحسينه ليكون مستقراً)
quality_mode = st.select_slider(
    "قوة الإزالة (كلما زادت القوة زاد وقت المعالجة)",
    options=["عادي", "قوي", "فائق (الأقوى)"],
    value="قوي"
)

if st.button("🚀 ابدأ العزل الآمن"):
    try:
        temp_dir = f"proc_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        if url_input:
            with st.spinner("جارٍ التحميل..."):
                source_path = download_video(url_input)
        elif uploaded_file:
            source_path = os.path.join(temp_dir, uploaded_file.name)
            with open(source_path, "wb") as f: f.write(uploaded_file.getbuffer())
        else:
            st.warning("يرجى تقديم ملف.")
            st.stop()

        with st.status("جارٍ الفصل (نظام توفير الذاكرة)...") as s:
            extractor = VocalExtractor()
            model = extractor.get_model()
            wav_input = os.path.join(temp_dir, "audio.wav")
            extractor.convert_to_wav(source_path, wav_input)
            
            wav, sr = torchaudio.load(wav_input)
            
            # ضبط الـ shifts بناءً على القوة المختارة
            shift_val = {"عادي": 1, "قوي": 5, "فائق (الأقوى)": 10}[quality_mode]

            # --- سر منع الانهيار: إضافة خاصية الـ split و segment ---
            # نقوم بتقسيم الصوت لقطع صغيرة (10 ثوانٍ) لمعالجتها دون استهلاك الـ RAM
            with torch.no_grad():
                sources = apply_model(
                    model, 
                    wav.unsqueeze(0), 
                    shifts=shift_val, 
                    split=True, 
                    overlap=0.25, 
                    device=extractor.device,
                    progress=True # إظهار التقدم في السجلات
                )[0]

            vocals = sources[model.sources.index("vocals")].cpu()
            
            # تنظيف الذاكرة فوراً بعد العملية
            del sources, wav
            gc.collect() 

            vocals_wav = os.path.join(temp_dir, "vocals.wav")
            torchaudio.save(vocals_wav, vocals, sr)
            s.update(label="اكتمل العزل بنجاح!", state="complete")

        # الإنتاج النهائي
        final_mp3 = "final_vocal.mp3"
        subprocess.run(["ffmpeg", "-i", vocals_wav, "-ac", "2", "-b:a", "192k", "-y", final_mp3], check=True, capture_output=True)
        
        st.audio(final_mp3)
        with open(final_mp3, "rb") as f:
            st.download_button("📥 تحميل الصوت الصافي", f, file_name=f"vocal_{int(time.time())}.mp3")

    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
    finally:
        # حذف الملفات المؤقتة لعدم ملء السيرفر
        if os.path.exists("cookies.txt"): os.remove("cookies.txt")
