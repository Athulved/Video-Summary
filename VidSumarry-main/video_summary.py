import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

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

# Save summary video
def save_summary_video(video_path, summary, output_path="summary_video.mp4", frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame is part of the summary
        if frame_id < len(summary) and summary[frame_id] == 1:
            out.write(frame)  # Write the frame to the output video

        frame_id += 1

    cap.release()
    out.release()
    print(f"Summary video saved to: {output_path}")

# Main function
def main():
    # Load the trained model
    input_size = 2048  # Feature size for ResNet-50
    hidden_size = 128
    model = VideoSummarizationModel(input_size, hidden_size)
    
    try:
        # Load the model onto the CPU
        model.load_state_dict(torch.load("video_summarization_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Model file not found. Please ensure 'video_summarization_model.pth' exists.")
        return
    
    model.eval()

    # Path to the video
    video_path = "/Users/mymac/Projects/VidSumarry/testVideo.mp4"

    # Generate summary
    summary = generate_summary(video_path, model)
    print("Generated summary:", summary)

    # Save summary video
    save_summary_video(video_path, summary, output_path="summary_video.mp4")

if __name__ == "__main__":
    main()