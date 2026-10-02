
import numpy as np
from scipy.signal import butter, filtfilt

SFREQ = 100  # 100 samples per second


def bandpass_filter(signal, sfreq=SFREQ, l_freq=0.3, h_freq=35, order=4):
    """
 apply a butterworth bandpass filter to the signal
l_freq removes very slow changes below the chosen frequency
h_freq removes high frequency noise above the chosen frequency
order controls how sharp the filter is
filtfilt applies the filter in both directions without shifting the eeg signal in time
    """
    nyquist = sfreq / 2
    low = l_freq / nyquist
    high = h_freq / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def filter_epochs(epochs, sfreq=SFREQ, l_freq=0.3, h_freq=35):
    """Apply bandpass_filter to every epoch in a (n_epochs, n_samples) array."""
    return np.array([bandpass_filter(e, sfreq, l_freq, h_freq) for e in epochs])


STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # fold N4 into N3 
    "Sleep stage R": "REM",
}


def load_and_epoch_real_recording(psg_path, hypnogram_path, eeg_channel="EEG Fpz-Cz",
                                   eog_channel="EOG horizontal",
                                   epoch_sec=30, l_freq=0.3, h_freq=35):
    """
    load one real sleep edf psg and hypnogram pair, filter the signals, 
    and split them into labeled 30 second epochs with eeg and eog channels.
    eog is important because eye movements are a key part of identifying rem sleep.
    returns epochs with eeg and eog data, and labels for wake, n1, n2, n3, and rem.
    """
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick([eeg_channel, eog_channel])
    raw.filter(l_freq, h_freq, verbose=False)

    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)

    sfreq = raw.info["sfreq"]
    all_data = raw.get_data()  # shape (2, n_samples): row 0 = EEG, row 1 = EOG

    epochs_list, labels_list = [], []
    for ann in raw.annotations:
        stage = STAGE_MAP.get(ann["description"])
        if stage is None:
            continue

        block_start = int(ann["onset"] * sfreq)
        block_end = block_start + int(ann["duration"] * sfreq)
        n_epoch_samples = int(epoch_sec * sfreq)

        for start in range(block_start, block_end, n_epoch_samples):
            end = start + n_epoch_samples
            if end > all_data.shape[1] or end > block_end:
                break
            epochs_list.append(all_data[:, start:end])  # (2, n_epoch_samples)
            labels_list.append(stage)

    return np.array(epochs_list), np.array(labels_list)


def trim_wake_periods(epochs, labels, buffer_epochs=60):
    """
    remove extra wake epochs from before and after the main sleep period.
    the recordings are much longer than the actual sleep time, so we keep
    only a small amount of wake time around the sleep period.
    wake epochs in between sleep stages are kept because they are part of
    the actual sleep recording.
    returns the trimmed epochs and labels.
    """
    non_wake_idx = np.where(labels != "Wake")[0]
    if len(non_wake_idx) == 0:
        return epochs, labels

    sleep_start = non_wake_idx[0]
    sleep_end = non_wake_idx[-1]

    trim_start = max(0, sleep_start - buffer_epochs)
    trim_end = min(len(labels), sleep_end + buffer_epochs + 1)

    return epochs[trim_start:trim_end], labels[trim_start:trim_end]