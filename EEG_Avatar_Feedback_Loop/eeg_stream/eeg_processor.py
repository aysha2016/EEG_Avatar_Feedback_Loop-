import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from scipy.signal import welch
import torch
import torch.nn as nn
import logging
import json
from pathlib import Path

class EEGProcessor:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.initialize_board()
        self.initialize_model()
        
    def _load_config(self, config_path):
        default_config = {
            'board_id': BoardIds.CYTON_BOARD.value,
            'serial_port': '/dev/ttyUSB0',
            'sample_rate': 250,
            'window_size': 4,
            'overlap': 0.5,
            'bands': {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            },
            'attention_threshold': 0.6
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
                logging.FileHandler('eeg_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EEGProcessor')

    def initialize_board(self):
        try:
            params = BrainFlowInputParams()
            params.serial_port = self.config['serial_port']
            self.board = BoardShim(self.config['board_id'], params)
            self.board.prepare_session()
            self.logger.info("EEG Board initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize board: {str(e)}")
            raise

    def initialize_model(self):
        self.model = AttentionCNN()
        model_path = Path('models/attention_cnn.pth')
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path))
            self.logger.info("Loaded pre-trained attention model")
        self.model.eval()

    def preprocess_signal(self, eeg_data):
        """Apply preprocessing steps to raw EEG data"""
        for channel in range(eeg_data.shape[0]):
            # Remove DC offset
            DataFilter.detrend(eeg_data[channel], DetrendOperations.CONSTANT.value)
            
            # Apply bandpass filter
            DataFilter.perform_bandpass(
                eeg_data[channel],
                self.config['sample_rate'],
                2.0, 45.0, 4,
                FilterTypes.BUTTERWORTH.value,
                0
            )
            
            # Remove noise
            DataFilter.perform_wavelet_denoising(
                eeg_data[channel],
                'db4',
                3
            )
        
        return eeg_data

    def extract_features(self, eeg_data):
        """Extract frequency band powers and other relevant features"""
        features = []
        
        for channel in range(eeg_data.shape[0]):
            channel_features = {}
            
            # Calculate power spectral density
            freqs, psd = welch(eeg_data[channel],
                             fs=self.config['sample_rate'],
                             nperseg=self.config['sample_rate'])
            
            # Extract band powers
            for band_name, (low, high) in self.config['bands'].items():
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.mean(psd[mask])
                channel_features[f"{band_name}_power"] = band_power
            
            # Calculate additional features
            channel_features['mean'] = np.mean(eeg_data[channel])
            channel_features['std'] = np.std(eeg_data[channel])
            channel_features['kurtosis'] = scipy.stats.kurtosis(eeg_data[channel])
            
            features.append(channel_features)
        
        return features

    def predict_attention(self, features):
        """Use CNN model to predict attention level"""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            attention_score = self.model(x)
            return attention_score.item()

    def get_cognitive_state(self, duration=10):
        """Main method to get cognitive state from EEG"""
        try:
            self.board.start_stream()
            self.logger.info(f"Starting EEG stream for {duration} seconds...")
            
            # Collect data
            data = self.board.get_board_data()
            self.board.stop_stream()
            
            # Process EEG data
            eeg_channels = self.board.get_eeg_channels(self.config['board_id'])
            eeg_data = data[eeg_channels]
            
            # Preprocess
            cleaned_data = self.preprocess_signal(eeg_data)
            
            # Extract features
            features = self.extract_features(cleaned_data)
            
            # Predict attention
            attention_score = self.predict_attention(features)
            
            cognitive_state = {
                'attention_score': attention_score,
                'is_focused': attention_score > self.config['attention_threshold'],
                'band_powers': features,
                'timestamp': time.time()
            }
            
            self.logger.info(f"Cognitive state processed: attention_score={attention_score:.2f}")
            return cognitive_state
            
        except Exception as e:
            self.logger.error(f"Error in cognitive state processing: {str(e)}")
            raise
        finally:
            self.board.stop_stream()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'board'):
            self.board.release_session()
            self.logger.info("Released board session")

class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 61, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 61)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x 