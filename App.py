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
st.set_page_config(page_title="عازل الموسيقى الاحترافي", page_icon="🎙️")

class VocalExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @st.cache_resource
    def get_model(_self):
        # استخدام موديل htdemucs_6s وهو الأفضل لعزل الصوت البشري بدقة عالية
        return get_model("htdemucs_6s").to(_self.device)

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
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    if cookies_content:
        with open("cookies.txt", "w") as f: f.write(cookies_content)
        ydl_opts['cookiefile'] = "cookies.txt"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

st.title("🎙️ عازل الموسيقى (الوضع الفائق)")
st.markdown("تم تحديث المحرك ليعطي أقوى نتائج عزل للصوت البشري فقط.")

tab1, tab2 = st.tabs(["🔗 رابط", "📂 رفع ملف"])
source_path = None

with tab1:
    url_input = st.text_input("ضع رابط المقطع هنا")
with tab2:
    uploaded_file = st.file_uploader("اختر ملف", type=["mp3", "wav", "mp4", "m4a", "flac"])

# خيارات القوة الجديدة
quality_mode = st.select_slider(
    "اختر قوة إزالة الموسيقى",
    options=["سريع (عادي)", "قوي (احترافي)", "فائق (أقوى إزالة - بطيء)"],
    value="قوي (احترافي)"
)

output_type = st.radio("المخرج النهائي", ["صوت فقط (MP3)", "فيديو بدون موسيقى"], index=0)

if st.button("🚀 ابدأ المعالجة بالقوة القصوى"):
    try:
        temp_dir = f"proc_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)

        if url_input:
            with st.status("جارٍ التحميل...") as s:
                source_path = download_video(url_input)
                s.update(label="تم التحميل!", state="complete")
        elif uploaded_file:
            source_path = os.path.join(temp_dir, uploaded_file.name)
            with open(source_path, "wb") as f: f.write(uploaded_file.getbuffer())
        else:
            st.warning("يرجى تقديم ملف.")
            st.stop()

        with st.status("جارٍ العزل بالفصل الفائق...") as s:
            extractor = VocalExtractor()
            model = extractor.get_model()
            wav_input = os.path.join(temp_dir, "audio.wav")
            extractor.convert_to_wav(source_path, wav_input)
            
            wav, sr = torchaudio.load(wav_input)
            wav = wav.to(extractor.device)

            # تحديد قوة الإزالة بناءً على اختيارك
            # الوضع الفائق يستخدم 10 تدويرات (shifts) لضمان نظافة الصوت البشري تماماً
            if quality_mode == "فائق (أقوى إزالة - بطيء)":
                current_shifts = 10
            elif quality_mode == "قوي (احترافي)":
                current_shifts = 5
            else:
                current_shifts = 1

            sources = apply_model(model, wav.unsqueeze(0), shifts=current_shifts, split=True, overlap=0.25, device=extractor.device)[0]
            vocals = sources[model.sources.index("vocals")].cpu()
            
            vocals_wav = os.path.join(temp_dir, "vocals.wav")
            torchaudio.save(vocals_wav, vocals, sr)
            s.update(label="اكتمل العزل الفائق بنجاح!", state="complete")

        if output_type == "صوت فقط (MP3)":
            final_path = "vocal_only.mp3"
            subprocess.run(["ffmpeg", "-i", vocals_wav, "-ac", "2", "-b:a", "192k", "-y", final_path], check=True, capture_output=True)
            st.audio(final_path)
        else:
            final_path = "video_no_music.mp4"
            cmd = ["ffmpeg", "-i", source_path, "-i", vocals_wav, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-y", final_path]
            subprocess.run(cmd, check=True, capture_output=True)
            st.video(final_path)

        with open(final_path, "rb") as f:
            st.download_button("📥 تحميل النتيجة النهائية", f, file_name=f"cleaned_{int(time.time())}.{'mp3' if 'MP3' in output_type else 'mp4'}")

    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
