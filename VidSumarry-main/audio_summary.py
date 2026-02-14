# import subprocess
# import whisper
# from transformers import pipeline

# def extract_audio(video_path, audio_output_path):
#     command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_output_path} -y"
#     subprocess.call(command, shell=True)
#     print(f"Audio extracted and saved to {audio_output_path}")

# def transcribe_audio(audio_path):
#     model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy
#     result = model.transcribe(audio_path)
#     return result["text"]

# def summarize_text(text, max_length=150):
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
#     return summary[0]["summary_text"]

# def main():
#     video_path = r"summary_video.mp4"
#     audio_path = r"aud.mp3"
    
#     extract_audio(video_path, audio_path)
#     transcribed_text = transcribe_audio(audio_path)
#     print("Transcribed Text:", transcribed_text)
    
#     summary = summarize_text(transcribed_text)
#     print("Summarized Text:", summary)

# if __name__ == "__main__":
#     main()


import subprocess
import whisper
from transformers import pipeline
from gtts import gTTS
import os

def extract_audio(video_path, audio_output_path):
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_output_path} -y"
    subprocess.call(command, shell=True)
    print(f"Audio extracted and saved to {audio_output_path}")

def transcribe_audio(audio_path):
    model = whisper.load_model("base", device="cpu")  # Run on CPU to avoid CUDA issues
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text, max_length=150):
    text = text[:2000]  # Avoid exceeding model limits
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Force CPU mode
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def main():
    video_path = r"devu.mp4"
    audio_path = r"aud.mp3"
    
    extract_audio(video_path, audio_path)
    transcribed_text = transcribe_audio(audio_path)
    print("Transcribed Text:", transcribed_text)
    
    if len(transcribed_text)<25:
        summary="No text summary available for this video"
    else:
        summary = summarize_text(transcribed_text)
    tts = gTTS(text=summary, lang='en')
    #save audio
    tts.save("output.mp3")
    #os.system("afplay output.mp3")
    print("Summarized Text:", summary)

if __name__ == "__main__":
    main()
