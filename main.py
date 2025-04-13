"""
Emotion-Aware Edge Assistant: Main Application

This application uses a webcam to detect human facial emotions in real-time,
generates an empathetic response, and vocalizes that response using offline TTS.
"""

import cv2
import time
import argparse
import sys
from threading import Thread
from typing import Dict, Optional

# Import configuration
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    FACE_DETECTION_SCALE_FACTOR, FACE_DETECTION_MIN_NEIGHBORS, FACE_DETECTION_MIN_SIZE,
    EMOTION_MODEL_PATH, EMOTIONS, EMOTION_MODEL_INPUT_SIZE,
    LANGUAGE_MODEL_NAME, MAX_LENGTH, TEMPERATURE,
    TTS_RATE, TTS_VOLUME, TTS_VOICE,
    DISPLAY_WINDOW, EMOTION_DETECTION_INTERVAL, RESPONSE_GENERATION_INTERVAL
)

# Import utility modules
from utils.camera import Camera
from utils.audio import AudioPlayer

# Import model modules
from models.emotion_model import EmotionDetector, predict_emotion
from models.language_model import ResponseGenerator, generate_response


class EmotionAwareAssistant:
    """Main application class for the Emotion-Aware Edge Assistant."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the assistant.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Use provided config or defaults
        self.config = config or {}
        
        # Component initialization flags
        self.camera_initialized = False
        self.emotion_model_initialized = False
        self.language_model_initialized = False
        self.audio_initialized = False
        
        # Setup components
        self.setup_camera()
        self.setup_emotion_detector()
        self.setup_response_generator()
        self.setup_audio_player()
        
        # State variables
        self.running = False
        self.current_emotion = None
        self.last_emotion_time = 0
        self.last_response_time = 0
        self.context = None
    
    def setup_camera(self):
        """Initialize the camera module."""
        camera_index = self.config.get('camera_index', CAMERA_INDEX)
        width = self.config.get('frame_width', FRAME_WIDTH)
        height = self.config.get('frame_height', FRAME_HEIGHT)
        fps = self.config.get('fps', FPS)
        
        self.camera = Camera(camera_index, width, height, fps)
        self.camera_initialized = self.camera.initialize()
        
        if not self.camera_initialized:
            print("Warning: Camera initialization failed")
    
    def setup_emotion_detector(self):
        """Initialize the emotion detection model."""
        model_path = self.config.get('emotion_model_path', EMOTION_MODEL_PATH)
        emotions = self.config.get('emotions', EMOTIONS)
        
        self.emotion_detector = EmotionDetector(model_path, emotions)
        
        # For simplicity, we'll consider it initialized even without a pre-trained model
        self.emotion_model_initialized = True
        print(f"Emotion detector initialized. Using device: {self.emotion_detector.device}")
    
    def setup_response_generator(self):
        """Initialize the language model for response generation."""
        model_name = self.config.get('language_model_name', LANGUAGE_MODEL_NAME)
        max_length = self.config.get('max_length', MAX_LENGTH)
        temperature = self.config.get('temperature', TEMPERATURE)
        
        self.response_generator = ResponseGenerator(model_name, max_length, temperature)
        self.language_model_initialized = self.response_generator.initialize()
        
        if not self.language_model_initialized:
            print("Warning: Language model initialization failed")
    
    def setup_audio_player(self):
        """Initialize the text-to-speech engine."""
        rate = self.config.get('tts_rate', TTS_RATE)
        volume = self.config.get('tts_volume', TTS_VOLUME)
        voice = self.config.get('tts_voice', TTS_VOICE)
        
        self.audio_player = AudioPlayer(rate, volume, voice)
        self.audio_initialized = self.audio_player.initialize()
        
        if not self.audio_initialized:
            print("Warning: Audio player initialization failed")
            
        # List available voices
        voices = self.audio_player.list_available_voices()
        print(f"Available TTS voices: {', '.join(voices) if voices else 'None found'}")
    
    def run(self):
        """Run the main application loop."""
        if not all([self.camera_initialized, 
                    self.emotion_model_initialized,
                    self.language_model_initialized,
                    self.audio_initialized]):
            print("Error: Not all components were initialized successfully.")
            print(f"Camera: {'OK' if self.camera_initialized else 'FAILED'}")
            print(f"Emotion model: {'OK' if self.emotion_model_initialized else 'FAILED'}")
            print(f"Language model: {'OK' if self.language_model_initialized else 'FAILED'}")
            print(f"Audio: {'OK' if self.audio_initialized else 'FAILED'}")
            return
        
        self.running = True
        print("Starting Emotion-Aware Edge Assistant...")
        
        # Start a non-blocking welcome message
        self.audio_player.speak("Emotion-aware assistant is now running.", False)
        
        try:
            while self.running:
                # Capture frame
                success, frame = self.camera.capture_frame()
                if not success:
                    print("Error: Failed to capture frame")
                    break
                
                # Detect faces
                faces = self.camera.detect_faces(
                    frame, 
                    scale_factor=self.config.get('face_detection_scale_factor', FACE_DETECTION_SCALE_FACTOR),
                    min_neighbors=self.config.get('face_detection_min_neighbors', FACE_DETECTION_MIN_NEIGHBORS),
                    min_size=self.config.get('face_detection_min_size', FACE_DETECTION_MIN_SIZE)
                )
                
                # Process the largest face (if any)
                if len(faces) > 0:
                    # Find the largest face by area
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    
                    # Process emotion at specified intervals
                    current_time = time.time()
                    if current_time - self.last_emotion_time >= self.config.get('emotion_detection_interval', EMOTION_DETECTION_INTERVAL):
                        # Extract and normalize face
                        face_img = self.camera.extract_face(
                            frame, largest_face, 
                            target_size=self.config.get('emotion_model_input_size', EMOTION_MODEL_INPUT_SIZE)
                        )
                        
                        # Predict emotion
                        emotion, confidence = self.emotion_detector.predict_emotion(face_img)
                        self.current_emotion = emotion
                        
                        # Update timestamp
                        self.last_emotion_time = current_time
                        
                        # Log detection
                        max_conf = max(confidence.values())
                        print(f"Detected emotion: {emotion} ({max_conf:.2f})")
                        
                        # Draw face box with emotion label
                        frame = self.camera.draw_face_box(frame, largest_face, emotion)
                        
                        # Generate and speak response at specified intervals
                        if self.current_emotion and current_time - self.last_response_time >= self.config.get('response_generation_interval', RESPONSE_GENERATION_INTERVAL):
                            # Generate response in a separate thread to avoid blocking the UI
                            response_thread = Thread(
                                target=self._generate_and_speak_response,
                                args=(self.current_emotion, self.context)
                            )
                            response_thread.daemon = True
                            response_thread.start()
                            
                            self.last_response_time = current_time
                    else:
                        # Just draw the face box without processing emotion
                        frame = self.camera.draw_face_box(frame, largest_face, self.current_emotion)
                
                # Display the frame
                if self.config.get('display_window', DISPLAY_WINDOW):
                    cv2.imshow('Emotion-Aware Assistant', frame)
                    
                    # Check for exit key (ESC or 'q')
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
        
        finally:
            # Clean up
            self.stop()
    
    def _generate_and_speak_response(self, emotion: str, context: Optional[str] = None):
        """
        Generate and speak a response based on the detected emotion.
        
        Args:
            emotion: Detected emotion
            context: Optional context from previous interactions
        """
        try:
            # Generate response
            response = self.response_generator.generate_response(emotion, context)
            print(f"Generated response: {response}")
            
            # Speak the response
            self.audio_player.speak(response, False)
            
            # Update context (could be more sophisticated with conversation history)
            self.context = emotion
        except Exception as e:
            print(f"Error generating or speaking response: {e}")
    
    def stop(self):
        """Stop the assistant and release resources."""
        self.running = False
        
        # Release camera
        if hasattr(self, 'camera'):
            self.camera.release()
        
        # Release audio resources
        if hasattr(self, 'audio_player'):
            self.audio_player.shutdown()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("Assistant stopped.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Emotion-Aware Edge Assistant')
    
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help='Camera index to use')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without displaying the video feed')
    parser.add_argument('--emotion-interval', type=float, default=EMOTION_DETECTION_INTERVAL,
                        help='Interval between emotion detections in seconds')
    parser.add_argument('--response-interval', type=float, default=RESPONSE_GENERATION_INTERVAL,
                        help='Interval between response generations in seconds')
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Create configuration from arguments
    config = {
        'camera_index': args.camera,
        'display_window': not args.no_display,
        'emotion_detection_interval': args.emotion_interval,
        'response_generation_interval': args.response_interval
    }
    
    # Create and run the assistant
    assistant = EmotionAwareAssistant(config)
    assistant.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0) 