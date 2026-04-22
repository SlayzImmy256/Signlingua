"""Data preprocessing with MediaPipe"""

import cv2
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipePreprocessor:
    """Extract landmarks from videos using MediaPipe Holistic"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.holistic = None
        self.mp_holistic = None
        
        # Landmark dimensions
        self.pose_dim = 33 * 4  # 33 landmarks * (x, y, z, visibility)
        self.hand_dim = 21 * 3  # 21 landmarks * (x, y, z)
        self.face_dim = 468 * 3  # 468 landmarks * (x, y, z)
        self.total_dim = self.pose_dim + 2 * self.hand_dim + self.face_dim
        
        # Initialize MediaPipe
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Initialize or reinitialize MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or mp is None:
            logger.warning("MediaPipe not available - using dummy preprocessor")
            return
        
        try:
            # Close existing instance if any
            if self.holistic is not None:
                try:
                    self.holistic.close()
                except:
                    pass
            
            # Create new instance
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=1,
                static_image_mode=False,
                smooth_landmarks=True
            )
            logger.info("MediaPipe initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self.holistic = None
            self.mp_holistic = None
            raise RuntimeError(f"MediaPipe initialization failed: {e}")
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single frame
        
        Args:
            frame: RGB image frame
            
        Returns:
            Flattened landmark array of shape (total_dim,) or None if detection fails
        """
        if self.holistic is None:
            logger.error("MediaPipe not initialized! Call _initialize_mediapipe() first")
            raise RuntimeError("MediaPipe not initialized. Please check MediaPipe installation.")
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Process frame
        results = self.holistic.process(frame_rgb)
        
        # Extract landmarks
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                           for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(self.pose_dim)
        landmarks.append(pose)
        
        # Left hand landmarks (21 * 3 = 63)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] 
                                for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand = np.zeros(self.hand_dim)
        landmarks.append(left_hand)
        
        # Right hand landmarks (21 * 3 = 63)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] 
                                 for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand = np.zeros(self.hand_dim)
        landmarks.append(right_hand)
        
        # Face landmarks (468 * 3 = 1404) - optional, can be excluded for efficiency
        # Uncomment if you want to include face landmarks
        # if results.face_landmarks:
        #     face = np.array([[lm.x, lm.y, lm.z] 
        #                    for lm in results.face_landmarks.landmark]).flatten()
        # else:
        #     face = np.zeros(self.face_dim)
        # landmarks.append(face)
        
        # Concatenate all landmarks
        landmarks_array = np.concatenate(landmarks)
        
        return landmarks_array
    
    def extract_landmarks_from_video(self, 
                                    video_path: str, 
                                    max_frames: int = 64,
                                    target_fps: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract landmarks from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            target_fps: Target FPS for frame sampling (None = use all frames)
            
        Returns:
            Landmark sequence of shape (num_frames, total_dim) or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if target_fps and target_fps < fps:
            frame_skip = int(fps / target_fps)
        else:
            frame_skip = 1
        
        landmarks_sequence = []
        frame_idx = 0
        
        while len(landmarks_sequence) < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames
            if frame_idx % frame_skip == 0:
                landmarks = self.extract_landmarks_from_frame(frame)
                
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        if not landmarks_sequence:
            logger.warning(f"No landmarks extracted from: {video_path}")
            return None
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks_sequence)
        
        # Pad or truncate to max_frames
        if len(landmarks_array) < max_frames:
            # Pad with zeros
            padding = np.zeros((max_frames - len(landmarks_array), landmarks_array.shape[1]))
            landmarks_array = np.vstack([landmarks_array, padding])
        else:
            # Truncate
            landmarks_array = landmarks_array[:max_frames]
        
        return landmarks_array
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to zero mean and unit variance
        
        Args:
            landmarks: Landmark array of shape (num_frames, total_dim)
            
        Returns:
            Normalized landmarks
        """
        # Calculate mean and std (excluding zero-padded frames)
        non_zero_mask = np.any(landmarks != 0, axis=1)
        
        if np.sum(non_zero_mask) > 0:
            mean = landmarks[non_zero_mask].mean(axis=0)
            std = landmarks[non_zero_mask].std(axis=0) + 1e-8
            
            # Normalize
            landmarks_normalized = landmarks.copy()
            landmarks_normalized[non_zero_mask] = (landmarks[non_zero_mask] - mean) / std
            
            return landmarks_normalized
        
        return landmarks
    
    def close(self):
        """Explicitly close MediaPipe resources"""
        if self.holistic is not None:
            try:
                self.holistic.close()
                logger.info("MediaPipe closed successfully")
            except Exception as e:
                logger.warning(f"Error closing MediaPipe: {e}")
            finally:
                self.holistic = None
    
    def __del__(self):
        """Cleanup"""
        self.close()


def preprocess_video(video_path: str, 
                    max_frames: int = 64,
                    normalize: bool = True) -> Optional[np.ndarray]:
    """
    Convenience function to preprocess a single video
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames
        normalize: Whether to normalize landmarks
        
    Returns:
        Preprocessed landmarks array
    """
    preprocessor = MediaPipePreprocessor()
    landmarks = preprocessor.extract_landmarks_from_video(video_path, max_frames)
    
    if landmarks is not None and normalize:
        landmarks = preprocessor.normalize_landmarks(landmarks)
    
    return landmarks
