import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os

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
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    features = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at the specified frame rate
        if frame_id % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # Preprocess and extract features
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
        self.fc = nn.Linear(hidden_size * 2, 1)  # Output importance score for each frame

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm)
        return torch.sigmoid(out)

# Generate summary
def generate_summary(video_path, model, frame_rate=1):
    # Extract features from the video
    features = extract_features(video_path, frame_rate)
    
    # Convert features to a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)

    # Generate importance scores
    model.eval()
    with torch.no_grad():
        scores = model(features).squeeze().numpy()

    # Select top-k frames as the summary
    k = int(0.1 * len(scores))  # Summarize 10% of the video
    top_k_indices = np.argsort(scores)[-k:]
    summary = np.zeros_like(scores)
    summary[top_k_indices] = 1

    return summary

# Split video into 4 parts
def split_video(video_path, output_dir, num_parts=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_part = total_frames // num_parts

    for i in range(num_parts):
        output_path = os.path.join(output_dir, f"part_{i+1}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        for frame_id in range(frames_per_part):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Saved: {output_path}")
    
    cap.release()

# Combine summarized frames into a single video
def combine_summarized_frames(summarized_frames, output_path, fps=30):
    if not summarized_frames:
        raise ValueError("No summarized frames to combine.")
    
    height, width, _ = summarized_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame in summarized_frames:
        out.write(frame)
    
    out.release()
    print(f"Combined video saved: {output_path}")

# Main function
def main():
    # Load the trained model
    input_size = 2048  # Feature size for ResNet-50
    hidden_size = 128
    model = VideoSummarizationModel(input_size, hidden_size)
    model.load_state_dict(torch.load("video_summarization_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Path to the video
    video_path = "/Users/mymac/Projects/VidSumarry/testVideo.mp4"
    output_dir = "video_parts"
    summarized_frames = []

    # Split video into 4 parts
    split_video(video_path, output_dir, num_parts=4)

    # Summarize each part
    for i in range(1, 5):
        part_path = os.path.join(output_dir, f"part_{i}.mp4")
        summary = generate_summary(part_path, model)
        
        # Extract summarized frames
        cap = cv2.VideoCapture(part_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id < len(summary) and summary[frame_id] == 1:
                summarized_frames.append(frame)
            frame_id += 1
        cap.release()

    # Combine summarized frames into a single video
    combined_output_path = "combined_summary.mp4"
    combine_summarized_frames(summarized_frames, combined_output_path)

if __name__ == "__main__":
    main()