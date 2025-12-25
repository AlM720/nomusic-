import streamlit as st
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
import os
import subprocess
import shutil
import time
import yt_dlp

# إعداد الصفحة
st.set_page_config(page_title="عازل الموسيقى الذكي", page_icon="🎵")

class VocalExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @st.cache_resource
    def get_model(_self):
        return get_model("htdemucs_6s").to(_self.device)

    def convert_to_wav(self, input_path, output_path):
        subprocess.run(["ffmpeg", "-i", input_path, "-vn", "-ac", "2", "-ar", "44100", "-y", output_path], check=True, capture_output=True)

# دالة التحميل المتقدمة (مثل Seal)
def download_video(url):
    output_path = "downloaded_input.mp4"
    cookies_content = st.secrets.get("coce", "") # جلب الكوكيز من الأرقام السرية
    
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'nocheckcertificate': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    if cookies_content:
        with open("cookies.txt", "w") as f: f.write(cookies_content)
        ydl_opts['cookiefile'] = "cookies.txt"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# واجهة المستخدم (نفس ميزات هاجنج فيس)
st.title("🎵 عازل الموسيقى والذكاء الاصطناعي")

tab1, tab2 = st.tabs(["🔗 رابط (يوتيوب/تيك توك)", "📂 رفع ملف مباشر"])
source_path = None

with tab1:
    url_input = st.text_input("ضع الرابط هنا")
with tab2:
    uploaded_file = st.file_uploader("اختر ملف", type=["mp3", "wav", "mp4", "m4a", "flac"])

quality_mode = st.radio("الجودة", ["أسرع (عادي)", "إزالة أقوى (أبطأ)"], index=1)
output_type = st.radio("نوع النتيجة النهائية", ["صوت", "فيديو"], index=0)

if st.button("🚀 ابدأ المعالجة الآن"):
    try:
        temp_dir = f"proc_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        # 1. جلب الملف
        if url_input:
            with st.status("جارٍ تحميل المقطع...") as s:
                source_path = download_video(url_input)
                s.update(label="تم التحميل بنجاح!", state="complete")
        elif uploaded_file:
            source_path = os.path.join(temp_dir, uploaded_file.name)
            with open(source_path, "wb") as f: f.write(uploaded_file.getbuffer())
        else:
            st.warning("يرجى تقديم ملف أو رابط.")
            st.stop()

        # 2. المعالجة
        with st.status("جارٍ فصل الموسيقى...") as s:
            extractor = VocalExtractor()
            model = extractor.get_model()
            
            # تحويل لـ WAV
            wav_input = os.path.join(temp_dir, "audio.wav")
            extractor.convert_to_wav(source_path, wav_input)
            
            # العزل
            wav, sr = torchaudio.load(wav_input)
            wav = wav.to(extractor.device)
            shifts = 0 if quality_mode == "أسرع (عادي)" else 5
            
            sources = apply_model(model, wav.unsqueeze(0), shifts=shifts, split=True, overlap=0.25, device=extractor.device)[0]
            vocals = sources[model.sources.index("vocals")].cpu()
            
            vocals_wav = os.path.join(temp_dir, "vocals.wav")
            torchaudio.save(vocals_wav, vocals, sr)
            s.update(label="تم الفصل بنجاح!", state="complete")

        # 3. الإنتاج النهائي
        if output_type == "صوت":
            final_path = "result.mp3"
            subprocess.run(["ffmpeg", "-i", vocals_wav, "-ac", "2", "-b:a", "192k", "-y", final_path], check=True, capture_output=True)
            st.audio(final_path)
        else:
            final_path = "result.mp4"
            # [span_4](start_span)دمج الصوت المعزول مع الفيديو الأصلي[span_4](end_span)
            cmd = ["ffmpeg", "-i", source_path, "-i", vocals_wav, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", final_path]
            subprocess.run(cmd, check=True, capture_output=True)
            st.video(final_path)

        with open(final_path, "rb") as f:
            st.download_button(f"📥 تحميل الـ{output_type}", f, file_name=f"no_music_{int(time.time())}.{'mp3' if output_type=='صوت' else 'mp4'}")

    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
    finally:
        if os.path.exists("cookies.txt"): os.remove("cookies.txt")
