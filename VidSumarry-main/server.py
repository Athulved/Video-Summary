


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import torch
import yt_dlp
import subprocess
import whisper
from gtts import gTTS
from transformers import pipeline
from ThreadSum import GenerateSummary, VideoSummarizationModel
import shutil
from groq import Groq
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Folder configurations
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'summaries'
TEXT_SUMMARIES_FOLDER = 'text_summaries'
AUDIO_SUMMARIES_FOLDER = 'audio_summaries'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEXT_SUMMARIES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_SUMMARIES_FOLDER, exist_ok=True)

# Load trained video summarization model
input_size = 2048
hidden_size = 128
model = VideoSummarizationModel(input_size, hidden_size)
model.load_state_dict(torch.load("video_summarization_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load Whisper Model (CPU Mode)
whisper_model = whisper.load_model("base", device="cpu")

# Initialize Groq client
try:
    client = Groq(api_key="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")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    client = None

def sanitize_filename(filename):
    """Sanitize filename to remove special characters"""
    return re.sub(r'[^\w\.-]', '_', filename)

def download_youtube_video(url, output_folder):
    """Download YouTube video using yt-dlp"""
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info_dict)

def extract_audio(video_path, audio_output_path):
    """Extract audio from video using ffmpeg with auto-overwrite"""
    command = [
        'ffmpeg',
        '-y',  # Auto-overwrite without prompt
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_output_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_text_with_groq(text):
    """Generate text summary using Groq API"""
    if not client:
        return "Summary service unavailable"
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "user",
                "content": f"Summarize the following text in plain English:\n\n{text}"
            }],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "No summary available due to an error."

def generate_audio_summary(text, audio_output_path):
    """Generate audio summary using gTTS"""
    tts = gTTS(text=text, lang='en')
    tts.save(audio_output_path)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        # Get and validate compression value (20-80 range)
        compression = float(request.form.get('compression', 20))
        compression = max(20, min(80, compression))
        compression_ratio = compression / 100  # Convert to decimal
        
        logger.info(f"Processing with compression: {compression}%")

        # Handle file upload or YouTube link
        if 'video' in request.files:
            video = request.files['video']
            sanitized_filename = sanitize_filename(video.filename)
            video_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
            video.save(video_path)
        elif 'youtube_link' in request.form:
            youtube_link = request.form['youtube_link']
            video_path = download_youtube_video(youtube_link, UPLOAD_FOLDER)
            sanitized_filename = os.path.basename(video_path)
        else:
            return jsonify({'error': 'No video file or YouTube link provided'}), 400

        # Generate unique filenames with compression percentage
        base_name = os.path.splitext(sanitized_filename)[0]
        output_suffix = f"_compressed_{int(compression)}"
        
        # Generate summarized video
        output_video = f"summary{output_suffix}_{sanitized_filename}"
        output_video_path = os.path.join(OUTPUT_FOLDER, output_video)
        GenerateSummary(video_path, compression_ratio, output_video_path)

        # Process text summary
        audio_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.mp3")
        extract_audio(video_path, audio_path)
        transcribed_text = transcribe_audio(audio_path)

        # Generate text summary
        summarized_text = summarize_text_with_groq(transcribed_text) if len(transcribed_text) >= 25 else "No text summary available"

        # Save text summary
        text_summary_path = os.path.join(TEXT_SUMMARIES_FOLDER, f"summary{output_suffix}_{base_name}.txt")
        with open(text_summary_path, "w", encoding="utf-8") as f:
            f.write(summarized_text)

        # Generate audio summary
        audio_summary_path = os.path.join(AUDIO_SUMMARIES_FOLDER, f"summary{output_suffix}_{base_name}.mp3")
        generate_audio_summary(summarized_text, audio_summary_path)

        return jsonify({
            'summary_video': output_video,
            'summary_text': summarized_text,
            'summary_audio': f"summary{output_suffix}_{base_name}.mp3",
            'compression_used': compression
        })

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summaries/<path:filename>')
def get_summary_video(filename):
    """Serve summarized video files"""
    return send_from_directory(OUTPUT_FOLDER, filename, mimetype='video/mp4')

@app.route('/text_summaries/<path:filename>')
def get_summary_text(filename):
    """Serve text summary files"""
    return send_from_directory(TEXT_SUMMARIES_FOLDER, filename, mimetype='text/plain')

@app.route('/audio_summaries/<path:filename>')
def get_summary_audio(filename):
    """Serve audio summary files"""
    return send_from_directory(AUDIO_SUMMARIES_FOLDER, filename, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(debug=True)