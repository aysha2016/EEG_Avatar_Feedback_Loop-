def detect_overload(eeg_data):
    attention_level = np.mean(eeg_data[0])
    if attention_level < 0.5:
        return True
    return False