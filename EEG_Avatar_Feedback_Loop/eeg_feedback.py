import numpy as np
import random
from avatar_generate import generate_avatar
from utils.filters import detect_overload
from utils.overlay import show_emotion
from IPython.display import Video

# Simulate EEG data for testing
def simulate_eeg_data():
    eeg_data = np.random.rand(8, 300)  # 8 channels, 300 data points (adjust as necessary)
    return eeg_data

def check_for_overload(eeg_data):
    attention_level = np.mean(eeg_data[0])  # Example: Use the first EEG channel
    if attention_level < 0.5:
        return True
    return False

def run_feedback_loop(mock_data=False, video_input="media/input_video.mp4"):
    if mock_data:
        eeg_data = simulate_eeg_data()
    else:
        eeg_data = get_real_eeg_data()

    if check_for_overload(eeg_data):
        print("⚠️ Overload detected! Triggering avatar feedback...")
        show_emotion("confused")
        generate_avatar(video_input)
    else:
        print("✅ User is focused.")

    display(Video("output/avatar_talk.mp4", embed=True))

run_feedback_loop(mock_data=True, video_input="media/input_video.mp4")