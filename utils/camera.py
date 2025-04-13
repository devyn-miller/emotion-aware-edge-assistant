"""
Camera utilities for the Emotion-Aware Edge Assistant.
"""
import cv2
import numpy as np
import time
from typing import Tuple, Optional, List

class Camera:
    """Camera handler for capturing and processing video frames."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the camera handler.
        
        Args:
            camera_index: Index of the camera to use
            width: Frame width
            height: Frame height
            fps: Desired frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.face_cascade = None
        
    def initialize(self) -> bool:
        """
        Initialize camera and load face detection model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Initialize webcam
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            self.release()
            return False
            
        return True
        
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
            
        success, frame = self.cap.read()
        return success, frame
        
    def detect_faces(self, frame: np.ndarray, 
                     scale_factor: float = 1.1,
                     min_neighbors: int = 5, 
                     min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image frame
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible face size
            
        Returns:
            List of face rectangles as (x, y, w, h)
        """
        if self.face_cascade is None:
            return []
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces
        
    def extract_face(self, frame: np.ndarray, face: Tuple[int, int, int, int], 
                     target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """
        Extract and preprocess a face region for emotion detection.
        
        Args:
            frame: Input image frame
            face: Face rectangle as (x, y, w, h)
            target_size: Target size for the extracted face
            
        Returns:
            Preprocessed face image
        """
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        resized_face = cv2.resize(gray_face, target_size)
        
        # Normalize pixel values to [0, 1]
        normalized_face = resized_face / 255.0
        
        return normalized_face
        
    def draw_face_box(self, frame: np.ndarray, face: Tuple[int, int, int, int], 
                      emotion: str = None, color: Tuple[int, int, int] = (0, 255, 0), 
                      thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box and emotion label for a detected face.
        
        Args:
            frame: Input image frame
            face: Face rectangle as (x, y, w, h)
            emotion: Detected emotion (optional)
            color: Box color in BGR format
            thickness: Line thickness
            
        Returns:
            Frame with annotations
        """
        x, y, w, h = face
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Add emotion label if provided
        if emotion:
            cv2.putText(frame, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
        
        return frame
        
    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None 