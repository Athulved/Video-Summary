
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import subprocess  # Add this import for FFmpeg

# Load pre-trained ResNet-50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
resnet.eval()

# Preprocessing function
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame).unsqueeze(0)

# Extract features from a video
def extract_features(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_frame(frame)
            with torch.no_grad():
                feature = resnet(input_tensor)
            features.append(feature.squeeze().numpy())

        frame_id += 1

    cap.release()
    return np.array(features)

# VideoSummarizationModel definition
class VideoSummarizationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(VideoSummarizationModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm)
        return torch.sigmoid(out)

# Generate summary
def generate_summary(video_path, model, frame_rate=1):
    features = extract_features(video_path, frame_rate)
    features = torch.tensor(features, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(features).squeeze().numpy()

    k = int(0.1 * len(scores))  # Summarize 10% of the video
    top_k_indices = np.argsort(scores)[-k:]
    summary = np.zeros_like(scores)
    summary[top_k_indices] = 1

    return summary

# Visualize summary as a single video (updated to use FFmpeg)
def visualize_summary(video_path, summary, output_video_path="summary_video.mp4", frame_rate=30):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use FFmpeg to write the video with H.264 codec
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',  # Input format
            '-pix_fmt', 'bgr24',  # Pixel format
            '-s', f'{frame_width}x{frame_height}',  # Frame size
            '-r', str(frame_rate),  # Frame rate
            '-i', '-',  # Input from pipe
            '-vcodec', 'libx264',  # Use H.264 codec
            '-acodec', 'aac',  # Use AAC codec (optional, if audio is needed)
            '-movflags', '+faststart',  # Move metadata to the beginning for streaming
            output_video_path
        ]

        # Start FFmpeg process
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id < len(summary) and summary[frame_id] == 1:
                # Write the frame to FFmpeg's stdin
                ffmpeg_process.stdin.write(frame.tobytes())

            frame_id += 1

        # Release resources
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    except Exception as e:
        print(f"Error generating summary video: {e}")