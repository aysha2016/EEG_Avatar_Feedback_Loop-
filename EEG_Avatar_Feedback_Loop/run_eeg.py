from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import time

def run_eeg_stream():
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'  # Change this as per your system

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    print("🔴 Streaming EEG for 10 seconds...")
    time.sleep(10)
    board.stop_stream()

    data = board.get_board_data()
    board.release_session()

    eeg_channel = board.get_eeg_channels(BoardIds.CYTON_BOARD.value)[0]
    eeg_signal = data[eeg_channel, :]

    # Bandpass filter to isolate alpha waves (attention metric)
    DataFilter.perform_bandpass(eeg_signal, BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value), 
                                8.0, 12.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    attention_metric = np.mean(np.abs(eeg_signal))
    print(f"🧠 Alpha Power (Attention Metric): {attention_metric:.4f}")
    return attention_metric

if __name__ == "__main__":
    run_eeg_stream()
