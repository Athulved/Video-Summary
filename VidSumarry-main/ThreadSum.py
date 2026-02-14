import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import concurrent.futures
import gc
import time
from functools import lru_cache
import threading
import subprocess


def convert_to_browser_compatible(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac',
        '-strict', '-2', output_path
    ]
    subprocess.call(command)

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cache the model to avoid reloading
@lru_cache(maxsize=1)
def get_resnet_model():
    print("Loading ResNet model...")
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device)
    resnet.eval()
    return resnet

# Preprocessing transform (defined once)
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process frames in batches for better efficiency
def extract_features(video_path, frame_rate=1, batch_size=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    features = []
    frame_indices = []
    
    # Get the model (cached)
    resnet = get_resnet_model()
    
    frame_id = 0
    batch_frames = []
    batch_indices = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            batch_indices.append(frame_id)
            
            # Process in batches
            if len(batch_frames) >= batch_size:
                # Convert to tensor and process batch
                batch_tensors = torch.stack([preprocess_transform(f) for f in batch_frames]).to(device)
                
                with torch.no_grad():
                    batch_features = resnet(batch_tensors)
                
                features.extend(batch_features.squeeze().cpu().numpy())
                frame_indices.extend(batch_indices)
                
                # Clear batch
                batch_frames = []
                batch_indices = []
                
                # Free memory
                del batch_tensors
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        frame_id += 1
    
    # Process any remaining frames
    if batch_frames:
        batch_tensors = torch.stack([preprocess_transform(f) for f in batch_frames]).to(device)
        with torch.no_grad():
            batch_features = resnet(batch_tensors)
        features.extend(batch_features.squeeze().cpu().numpy())
        frame_indices.extend(batch_indices)
    
    cap.release()
    
    return np.array(features), np.array(frame_indices)

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

# Generate summary directly to output file
def generate_summary(video_path, model, summary_dict, ind, percentage, frame_rate=1):
    print(f"Processing part {ind+1}...")
    # Extract features from the video
    features, frame_indices = extract_features(video_path, frame_rate, batch_size=32)
    
    # Check if we have any features
    if len(features) == 0:
        print(f"Warning: No features extracted from {video_path}")
        return
    
    # Convert features to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Generate importance scores
    with torch.no_grad():
        scores = model(features_tensor).squeeze().cpu().numpy()

    # Select top-k frames as the summary
    k = max(1, int(percentage * len(scores)))  # Ensure at least 1 frame
    top_k_indices = np.argsort(scores)[-k:]
    summary = np.zeros_like(scores)
    summary[top_k_indices] = 1
    
    # Store summary info in shared dictionary
    summary_dict[ind] = (summary, frame_indices)
    
    print(f"Part {ind+1}: Found {sum(summary)} important frames out of {len(summary)} frames")
    
    # Clean up
    del features_tensor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return summary

# Split video into parts 
def split_video_ffmpeg(video_path, output_dir, num_parts=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_part = total_frames // num_parts
    cap.release()
    
    threads = []
    for i in range(num_parts):
        start_frame = i * frames_per_part
        end_frame = (i + 1) * frames_per_part if i != num_parts - 1 else total_frames
        output_path = os.path.join(output_dir, f"part_{i+1}.mp4")
        
        thread = threading.Thread(target=split_video_part, args=(video_path, output_path, start_frame, end_frame))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
        
def split_video_part(video_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    out.release()
    cap.release()
    print(f"Saved: {output_path}")
    
    print("Video successfully split into 4 parts.")
# Write summarized frames to output
def write_summarized_frames(video_path, frame_indices, summary, output_writer):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_id = 0
    frames_written = 0
    
    # Map frame indices to summary values for quick lookup
    summary_map = {frame_indices[i]: summary[i] for i in range(len(summary))}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be included in the summary
        if frame_id in summary_map and summary_map[frame_id] == 1:
            output_writer.write(frame)
            frames_written += 1
        
        frame_id += 1
    
    cap.release()
    print(f"Wrote {frames_written} frames from {video_path}")
    return frames_written

# Combine summarized parts into a single video
def create_summarized_video(video_parts, summaries_dict, output_path, fps=30):
    # Get video properties from the first part
    first_video = video_parts[0]
    cap = cv2.VideoCapture(first_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {first_video}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter("./summary_video.mp4", fourcc, fps, (width, height))
    
    total_frames_written = 0
    
    # Process each part one by one
    for i, video_path in enumerate(video_parts):
        if i not in summaries_dict:
            print(f"Warning: No summary data for part {i+1}")
            continue
        
        summary, frame_indices = summaries_dict[i]
        frames_written = write_summarized_frames(video_path, frame_indices, summary, out)
        total_frames_written += frames_written
    
    out.release()
    
    if total_frames_written > 0:
        convert_to_browser_compatible("./summary_video.mp4", output_path)
        print(f"Combined video saved: summary_video.mp4 with {total_frames_written} frames")
    else:
        print(f"ERROR: No frames were written to the output video!")

# Main function
def GenerateSummary(vid_path, percentage, out_path):
    start_time = time.time()

    file_to_remove = ["summary_video.mp4","part_1.mp4","part_2.mp4","part_3.mp4","part_4.mp4"]
    for file in file_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"'{file}' has been removed.")
        else:
            print(f"'{file}' does not exist in the current directory.")

    # Load the model ONCE
    input_size = 2048  # Feature size for ResNet-50
    hidden_size = 128
    model = VideoSummarizationModel(input_size, hidden_size)
    model.load_state_dict(torch.load("video_summarization_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Path to the video
    video_path = vid_path
    output_dir = "./"
    
    # Split the video
    split_video_ffmpeg(video_path, output_dir)
    video_parts = ["part_1.mp4","part_2.mp4","part_3.mp4","part_4.mp4"]
    # Shared dictionary for summaries
    summary_dict = {}
    
    # Process each part using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, part_path in enumerate(video_parts):
            future = executor.submit(generate_summary, part_path, model, summary_dict, i, percentage)
            futures.append(future)
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    # Check if we have any summaries
    if not summary_dict:
        print("Error: No summaries were generated!")
        return
    
    # Create the combined summary video
    output_path = out_path
    create_summarized_video(video_parts, summary_dict, output_path)
    
    # Verify the output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Output file size: {file_size} bytes")
        
        # Check if the file is empty
        if file_size == 0:
            print("Warning: Output file is empty!")
    else:
        print(f"Error: Output file {output_path} was not created!")
    
    # Execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")

    # Final cleanup
    del model
    get_resnet_model.cache_clear()  # Clear the cached model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()