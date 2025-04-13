"""
Audio utilities for the Emotion-Aware Edge Assistant.
"""
import pyttsx3
import threading
from typing import Optional, List

class AudioPlayer:
    """Text-to-speech handler for vocalizing assistant responses."""
    
    def __init__(self, rate: int = 150, volume: float = 1.0, voice: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
            voice: Specific voice to use, or None for system default
        """
        self.rate = rate
        self.volume = volume
        self.voice_name = voice
        self.engine = None
        self.is_speaking = False
        self.speech_thread = None
        
    def initialize(self) -> bool:
        """
        Initialize the TTS engine.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set specific voice if provided
            if self.voice_name is not None:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if self.voice_name in voice.name:
                        self.engine.setProperty('voice', voice.id)
                        break
            
            return True
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            return False
    
    def list_available_voices(self) -> List[str]:
        """
        Get a list of available voices.
        
        Returns:
            List of voice names
        """
        if self.engine is None:
            return []
            
        voices = self.engine.getProperty('voices')
        return [voice.name for voice in voices]
    
    def speak(self, text: str, block: bool = False):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            block: Whether to block until speaking is complete
        """
        if self.engine is None:
            return
            
        # Stop any current speech
        self.stop()
        
        if block:
            # Blocking call
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
        else:
            # Non-blocking call in a separate thread
            self.speech_thread = threading.Thread(target=self._speak_threaded, args=(text,))
            self.speech_thread.daemon = True
            self.speech_thread.start()
    
    def _speak_threaded(self, text: str):
        """
        Internal method to speak in a separate thread.
        
        Args:
            text: Text to speak
        """
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False
    
    def stop(self):
        """Stop any ongoing speech."""
        if self.engine is not None and self.is_speaking:
            self.engine.stop()
            self.is_speaking = False
            
    def shutdown(self):
        """Release TTS resources."""
        self.stop()
        # The pyttsx3 engine doesn't have an explicit cleanup method 