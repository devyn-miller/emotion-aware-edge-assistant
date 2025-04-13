"""
Emotion detection model for the Emotion-Aware Edge Assistant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import cv2

class EmotionCNN(nn.Module):
    """Convolutional Neural Network for emotion detection."""
    
    def __init__(self, num_emotions: int = 7):
        """
        Initialize the CNN model.
        
        Args:
            num_emotions: Number of emotion classes to predict
        """
        super(EmotionCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Calculate flattened size after convolutions
        # 48x48 -> 24x24 -> 12x12 -> 6x6 with 128 channels
        self.fc_input_size = 128 * 6 * 6
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_emotions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Logits for each emotion class
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EmotionDetector:
    """Handles emotion detection from facial images."""
    
    def __init__(self, model_path: Optional[str] = None, emotions: List[str] = None):
        """
        Initialize the emotion detector.
        
        Args:
            model_path: Path to the trained model weights, or None to use a new model
            emotions: List of emotion labels, or None to use defaults
        """
        if emotions is None:
            self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        else:
            self.emotions = emotions
            
        self.model = EmotionCNN(len(self.emotions))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.model.eval()
    
    def load_model(self, model_path: str) -> bool:
        """
        Load model weights from a file.
        
        Args:
            model_path: Path to the model weights file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save model weights to a file.
        
        Args:
            model_path: Path to save the model weights
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def preprocess_image(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face image for the model.
        
        Args:
            face_img: Grayscale face image (normalized to [0, 1])
            
        Returns:
            Tensor ready for model input
        """
        # Add batch and channel dimensions
        tensor = torch.from_numpy(face_img).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict the emotion from a face image.
        
        Args:
            face_img: Grayscale face image (normalized to [0, 1])
            
        Returns:
            Tuple of (predicted_emotion, confidence_scores)
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Preprocess the image
        tensor = self.preprocess_image(face_img)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get prediction
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
        predicted_emotion = self.emotions[predicted_idx]
        
        # Get confidence scores for all emotions
        confidence_scores = {emotion: float(probabilities[i].item()) 
                           for i, emotion in enumerate(self.emotions)}
        
        return predicted_emotion, confidence_scores
    
    def train(self, dataloader, num_epochs: int = 10, 
              learning_rate: float = 0.001) -> List[float]:
        """
        Train the emotion detection model.
        
        Args:
            dataloader: PyTorch DataLoader with training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            List of training losses
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        return losses


def predict_emotion(frame: np.ndarray, face_rect: Tuple[int, int, int, int], 
                   detector: EmotionDetector, target_size: Tuple[int, int] = (48, 48)) -> str:
    """
    Predict emotion from a frame with a detected face.
    
    Args:
        frame: Camera frame
        face_rect: Face rectangle as (x, y, w, h)
        detector: Initialized EmotionDetector
        target_size: Input size for the emotion model
        
    Returns:
        Predicted emotion label
    """
    x, y, w, h = face_rect
    face_img = frame[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    resized_face = cv2.resize(gray_face, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized_face = resized_face / 255.0
    
    # Predict emotion
    emotion, _ = detector.predict_emotion(normalized_face)
    
    return emotion 