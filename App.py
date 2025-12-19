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

class VocalExtractor:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"استخدام الجهاز: {self.device}")

    def get_model(self):
        if self.model is None:
            st.write("جارٍ تحميل نموذج Demucs...")
            self.model = get_model("htdemucs_6s").to(self.device)
        return self.model

    def convert_to_wav(self, input_path: str, output_path: str) -> None:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vn", "-ac", "2", "-ar", "44100",
            "-acodec", "pcm_s16le", "-y", output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def save_as_mp3(self, wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
        cmd = [
            "ffmpeg", "-i", wav_path,
            "-ac", "2", "-ar", "44100",
            "-b:a", bitrate, "-y", mp3_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def download_from_url(url):
    temp_dir = f"temp_download_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, "input.%(ext)s")
    ydl_opts = {
        'outtmpl': output_path,
        'quiet': True,
        'format': 'bestaudio[ext=m4a]',  # تنزيل صوت جاهز دون حاجة لدمج فيديو
        'ffmpeg_location': '/usr/bin/ffmpeg',  # مسار ffmpeg في Streamlit Cloud
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except Exception as e:
            st.error(f"خطأ في التنزيل: {str(e)}. جرب رابطًا آخر أو تحقق من الإنترنت.")
            return None, None
    downloaded_file = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))][0]
    return os.path.join(temp_dir, downloaded_file), temp_dir

def process_input(input_path, is_url, output_type, quality_mode):
    if not input_path:
        st.error("ارفع ملفًا أو أدخل رابطًا.")
        return None, None

    # إذا رابط، نزله أولاً
    download_dir = None
    if is_url:
        st.write("جارٍ تنزيل الملف من الرابط...")
        input_path, download_dir = download_from_url(input_path)
        if input_path is None:  # إذا فشل التنزيل
            return None, None

    # كشف نوع الملف
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext not in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]

    # إذا ملف صوتي، اجبر output_type على "صوت"
    if not is_video:
        output_type = "صوت"

    # تحديد shifts بناءً على الجودة
    shifts = 0 if quality_mode == "أسرع (جودة أقل)" else 5

    extractor = VocalExtractor()
    temp_dir = f"temp_proc_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        st.write("جارٍ الإعداد...")
        wav_path = os.path.join(temp_dir, "audio.wav")
        st.write("جارٍ التحويل (20%)...")
        extractor.convert_to_wav(input_path, wav_path)

        st.write("جارٍ تحميل النموذج (40%)...")
        model = extractor.get_model()

        st.write("جارٍ الفصل (60%)...")
        wav, sr = torchaudio.load(wav_path)
        wav = wav.to(extractor.device)

        sources = apply_model(
            model,
            wav.unsqueeze(0),
            shifts=shifts,
            split=True,
            overlap=0.25,
            device=extractor.device
        )[0]

        sources = sources.cpu()
        vocal_index = model.sources.index("vocals")
        vocals = sources[vocal_index]

        vocals_wav = os.path.join(temp_dir, "vocals.wav")
        torchaudio.save(vocals_wav, vocals, sr)

        st.write("جارٍ الحفظ (90%)...")
        os.makedirs("outputs", exist_ok=True)
        timestamp = int(time.time())

        if output_type == "صوت":
            output_path = os.path.join("outputs", f"vocals_{timestamp}.mp3")
            extractor.save_as_mp3(vocals_wav, output_path, bitrate="192k")
            return output_path, "audio"
        else:  # "فيديو"
            output_path = os.path.join("outputs", f"no_music_{timestamp}.mp4")
            cmd = [
                "ffmpeg", "-i", input_path,
                "-i", vocals_wav,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path, "video"

    except Exception as e:
        st.error(f"خطأ: {str(e)}")
        return None, None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if download_dir:
            shutil.rmtree(download_dir, ignore_errors=True)

# واجهة Streamlit
st.title("إزالة الموسيقى من الصوت أو الفيديو")
input_type = st.radio("اختر طريقة الإدخال", ["رفع ملف", "رابط (يوتيوب أو غيره)"])

input_path = None
is_url = False
if input_type == "رفع ملف":
    uploaded_file = st.file_uploader("ارفع ملف صوت أو فيديو", type=["mp3", "wav", "m4a", "flac", "aac", "ogg", "mp4", "mkv"])
    if uploaded_file:
        input_path = uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    url = st.text_input("أدخل رابط يوتيوب أو غيره")
    if url:
        input_path = url
        is_url = True

quality_mode = st.radio("اختر السرعة والجودة", ["أسرع (جودة أقل)", "جودة أعلى (أبطأ)"], index=1)

output_type = st.radio("نوع الإخراج", ["صوت", "فيديو"], index=0)

if st.button("إزالة الموسيقى"):
    output_path, output_format = process_input(input_path, is_url, output_type, quality_mode)
    if output_path:
        st.success("تم بنجاح!")
        if output_format == "audio":
            st.audio(output_path)
        else:
            st.video(output_path)