import os
import numpy as np
import scipy.io
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define paths
features_dir = "/Users/mymac/Projects/VidSumarry/Dataset/SUMme"
annotations_dir = "/Users/mymac/Projects/VidSumarry/Dataset/SUMme"

# Load pre-trained ResNet-50 for feature extraction
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
def extract_features(frames):
    features = []
    for frame in frames:
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            feature = resnet(input_tensor)
        features.append(feature.squeeze().numpy())
    return np.array(features)

# VideoSummarizationModel definition
class VideoSummarizationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(VideoSummarizationModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)  # Output importance score for each frame

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm)
        return torch.sigmoid(out)

# Initialize model
input_size = 2048  # Feature size for ResNet-50
hidden_size = 128
model = VideoSummarizationModel(input_size, hidden_size)
print(model)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with accuracy and loss tracking
def train_model(features, annotations, model, criterion, optimizer, num_epochs=50):
    # Prepare data
    X = features  # Input features
    y = annotations["gt_score"]  # Target labels

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        correct = (predicted == y_train).float().sum()
        accuracy = correct / y_train.shape[0]

        # Store training loss and accuracy
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            # Calculate validation accuracy
            val_predicted = (val_outputs > 0.5).float()
            val_correct = (val_predicted == y_val).float().sum()
            val_accuracy = val_correct / y_val.shape[0]

            # Store validation loss and accuracy
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss.item():.4f}, Train Acc: {accuracy.item():.4f}, "
              f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Process all feature and annotation files in the directory
for file_name in os.listdir(features_dir):
    if file_name.endswith(".npy"):
        # Load .npy file (all frames)
        features_path = os.path.join(features_dir, file_name)
        features = np.load(features_path)  # Shape: (N, 128, 128, 3)

        # Load corresponding .mat file (annotations)
        annotation_file_name = file_name.replace(".npy", ".mat")
        annotations_path = os.path.join(annotations_dir, annotation_file_name)
        annotations = scipy.io.loadmat(annotations_path)

        # Extract relevant annotations
        gt_score = annotations["gt_score"]  # Ground truth scores
        nFrames = annotations["nFrames"][0][0]  # Number of annotated frames

        # Calculate the step size to select key frames
        step_size = len(features) // nFrames

        # Select key frames from the features
        key_frames = features[::step_size][:nFrames]  # Shape: (nFrames, 128, 128, 3)

        # Verify alignment
        if len(key_frames) == nFrames:
            print(f"Processing {file_name}: Key frames and annotations are aligned.")
        else:
            print(f"Processing {file_name}: Mismatch in key frames and annotations.")
            continue

        # Extract features from key frames
        extracted_features = extract_features(key_frames)  # Shape: (nFrames, 2048)

        # Train the model
        print(f"Training model on {file_name}...")
        train_model(extracted_features, annotations, model, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), "video_summarization_model.pth")
print("Model saved as video_summarization_model.pth")