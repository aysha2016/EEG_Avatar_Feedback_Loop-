# EEG-Avatar Feedback Loop for Cochlear Implant Users

This project combines EEG data to trigger real-time avatar feedback loops. It aims to provide adaptive feedback based on EEG signals for Cochlear Implant users.

## Installation

Install dependencies in Colab or locally:

```bash
!pip install brainflow opencv-python gdown
```

### Running the Feedback Loop

```python
from eeg_feedback import run_feedback_loop
run_feedback_loop(mock_data=True, video_input="media/input_video.mp4")
```

This will simulate EEG data and trigger the avatar feedback loop. The avatar will be generated using Wav2Lip and emotion icons will be overlaid based on EEG input.

## License

This project is licensed under the MIT License.