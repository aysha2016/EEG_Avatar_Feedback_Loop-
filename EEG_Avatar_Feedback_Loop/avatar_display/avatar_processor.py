import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from moviepy.editor import VideoFileClip
import logging
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

class AvatarProcessor:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.initialize_models()
        self.setup_mediapipe()
        
    def _load_config(self, config_path):
        default_config = {
            'emotion_mapping': {
                'low_attention': '😴',
                'medium_attention': '😊',
                'high_attention': '🎯'
            },
            'caption_templates': {
                'low_attention': "Attention needed! Let's focus...",
                'medium_attention': "Good progress! Keep it up!",
                'high_attention': "Excellent focus! You're doing great!"
            },
            'video_width': 640,
            'video_height': 480,
            'overlay_opacity': 0.8
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('avatar_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AvatarProcessor')

    def initialize_models(self):
        """Initialize lip sync and emotion models"""
        self.lip_sync_model = LipSyncModel()
        self.emotion_model = EmotionModel()
        
        # Load pre-trained models if available
        model_dir = Path('models')
        if (model_dir / 'lip_sync.pth').exists():
            self.lip_sync_model.load_state_dict(
                torch.load(model_dir / 'lip_sync.pth')
            )
        if (model_dir / 'emotion.pth').exists():
            self.emotion_model.load_state_dict(
                torch.load(model_dir / 'emotion.pth')
            )

    def setup_mediapipe(self):
        """Initialize MediaPipe for face mesh detection"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame, audio_features=None):
        """Process a single frame for lip sync and emotion"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Extract lip landmarks
            lip_landmarks = self._extract_lip_landmarks(landmarks)
            
            # Generate lip sync if audio features available
            if audio_features is not None:
                lip_sync = self.lip_sync_model(
                    torch.tensor(lip_landmarks),
                    torch.tensor(audio_features)
                )
                frame = self._apply_lip_sync(frame, lip_sync, landmarks)
            
            # Apply emotion overlay
            emotion_features = self.emotion_model(torch.tensor(lip_landmarks))
            frame = self._apply_emotion_overlay(frame, emotion_features)
        
        return frame

    def _extract_lip_landmarks(self, landmarks):
        """Extract lip landmarks from face mesh"""
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        lip_points = []
        
        for idx in lip_indices:
            landmark = landmarks.landmark[idx]
            lip_points.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(lip_points)

    def _apply_lip_sync(self, frame, lip_sync, landmarks):
        """Apply lip sync transformations to the frame"""
        # Implementation of lip sync application
        # This would involve warping the mouth region based on
        # the predicted lip sync parameters
        return frame

    def _apply_emotion_overlay(self, frame, emotion_features):
        """Apply emotion-based overlay to the frame"""
        overlay = frame.copy()
        emotion_score = emotion_features.item()
        
        # Determine emotion state
        if emotion_score < 0.3:
            emotion = 'low_attention'
        elif emotion_score < 0.7:
            emotion = 'medium_attention'
        else:
            emotion = 'high_attention'
        
        # Add emoji and caption
        emoji = self.config['emotion_mapping'][emotion]
        caption = self.config['caption_templates'][emotion]
        
        # Convert to PIL for text overlay
        pil_image = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Add emoji
        font = ImageFont.truetype("arial.ttf", 60)
        draw.text((30, 30), emoji, font=font, fill=(255, 255, 255))
        
        # Add caption
        font = ImageFont.truetype("arial.ttf", 30)
        draw.text((30, 100), caption, font=font, fill=(255, 255, 255))
        
        # Convert back to OpenCV format
        overlay = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Blend with original frame
        return cv2.addWeighted(
            overlay,
            self.config['overlay_opacity'],
            frame,
            1 - self.config['overlay_opacity'],
            0
        )

    def process_video(self, video_path, cognitive_state):
        """Process entire video with cognitive state feedback"""
        try:
            clip = VideoFileClip(video_path)
            processed_frames = []
            
            for frame in clip.iter_frames():
                # Convert frame to BGR for OpenCV processing
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process frame with cognitive state
                processed_frame = self.process_frame(
                    frame_bgr,
                    cognitive_state.get('attention_score', 0.5)
                )
                
                processed_frames.append(processed_frame)
            
            # Create processed video clip
            processed_clip = VideoFileClip(video_path).set_frames(processed_frames)
            
            # Save processed video
            output_path = f"processed_{Path(video_path).name}"
            processed_clip.write_videofile(output_path)
            
            self.logger.info(f"Processed video saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources"""
        self.face_mesh.close()
        self.logger.info("Cleaned up avatar processor resources")

class LipSyncModel(nn.Module):
    def __init__(self):
        super(LipSyncModel, self).__init__()
        self.lstm = nn.LSTM(input_size=30, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 20)  # 20 lip sync parameters

    def forward(self, lip_landmarks, audio_features):
        # Combine lip landmarks and audio features
        x = torch.cat([lip_landmarks, audio_features], dim=-1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x 