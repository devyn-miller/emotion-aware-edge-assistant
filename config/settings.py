"""
Configuration settings for the Emotion-Aware Edge Assistant.
"""

# Camera settings
CAMERA_INDEX = 0  # Default camera (usually built-in webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Face detection settings
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (30, 30)

# Emotion detection model settings
EMOTION_MODEL_PATH = "models/emotion_model_weights.pth"  # Path to be used when saving/loading model
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_MODEL_INPUT_SIZE = (48, 48)  # Expected input size for emotion CNN

# Language model settings
LANGUAGE_MODEL_NAME = "distilgpt2"  # HuggingFace model name
MAX_LENGTH = 50  # Maximum length of generated text
TEMPERATURE = 0.7  # Higher values mean more creative responses

# Audio settings
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 1.0  # Volume level (0.0 to 1.0)
TTS_VOICE = None  # Default system voice

# Application settings
DISPLAY_WINDOW = True  # Whether to show the camera feed window
EMOTION_DETECTION_INTERVAL = 1.0  # Seconds between emotion detections
RESPONSE_GENERATION_INTERVAL = 3.0  # Seconds between response generations 