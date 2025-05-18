# EEG-Based Avatar Feedback System

A real-time AI system that detects cognitive load from EEG signals and provides multimodal feedback through a lip-synced avatar.

## Features

- Real-time EEG signal processing and cognitive load detection
- Deep learning-based attention classification using CNNs and LSTMs
- Lip-synced avatar with emotion-based feedback
- Configurable feedback thresholds and display settings
- Comprehensive logging and metrics tracking

## System Architecture

### EEG Processing
- Signal acquisition from Cyton board
- Real-time preprocessing (filtering, denoising)
- Feature extraction (time and frequency domain)
- Deep learning-based attention classification

### Avatar Processing
- Real-time face mesh detection
- LSTM-based lip synchronization
- Emotion overlay system
- Configurable feedback display

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the system:
- Edit `config/config.json` for general settings
- Adjust `config/eeg_config.json` for EEG processing parameters
- Modify `config/avatar_config.json` for avatar display settings

3. Prepare models:
- Place trained models in the `models/` directory:
  - `attention_cnn.pth`: EEG attention classification model
  - `lip_sync.pth`: Avatar lip synchronization model
  - `emotion.pth`: Emotion expression model

## Usage

Run the system with default configuration:
```bash
python main.py
```

Use custom configuration:
```bash
python main.py --config path/to/config.json
```

## Configuration

### EEG Settings
- Board configuration (ID, port, sampling rate)
- Signal preprocessing parameters
- Feature extraction settings
- Model configuration

### Avatar Settings
- Video properties (dimensions, FPS)
- Emotion mappings and captions
- Overlay appearance and positioning
- Face mesh and lip sync parameters

## Metrics and Logging

The system maintains comprehensive logs and metrics:
- Session metrics stored in `metrics/`
- System logs in `logs/`
- Real-time performance monitoring

## Development

### Project Structure
```
.
├── config/
│   ├── config.json
│   ├── eeg_config.json
│   └── avatar_config.json
├── eeg_stream/
│   └── eeg_processor.py
├── avatar_display/
│   └── avatar_processor.py
├── models/
│   ├── attention_cnn.pth
│   ├── lip_sync.pth
│   └── emotion.pth
├── main.py
└── requirements.txt
```

### Key Components
- `EEGProcessor`: Handles EEG signal processing and classification
- `AvatarProcessor`: Manages avatar display and feedback
- `FeedbackSystem`: Integrates EEG and avatar components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BrainFlow for EEG data acquisition
- MediaPipe for face mesh detection
- PyTorch for deep learning models
- OpenCV for video processing 