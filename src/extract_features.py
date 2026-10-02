"""
turn each 30s eeg epoch into 4 simple band-power features.
we use welch's method to find the power across different frequencies,
then calculate how much power is in the delta, theta, alpha and beta bands.
this gives us 4 useful numbers instead of the original 3000 samples,
while keeping information that helps distinguish sleep stages.
"""
import numpy as np
from scipy.signal import welch

SFREQ = 100

# gamma intentionally excluded 
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def band_power(epoch, sfreq=SFREQ, bands=BANDS, relative=True):
    """
calculate the power in each frequency band for one epoch.
by default, relative=true converts each band's power into a percentage
of the total power across all bands.
This is useful because:
1 It reduces the effect of artifacts like eye blinks, which can
   artificially increase delta power, especially in frontal channels.
2 It makes eeg power more comparable between different people,
   since their raw signal amplitudes can vary for reasons unrelated
   to sleep stage.

returns a dictionary: {band_name: power_value}.
    """
    freqs, psd = welch(epoch, fs=sfreq, nperseg=sfreq * 4)  # 4s windows

    raw_powers = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        raw_powers[band_name] = np.trapezoid(psd[mask], freqs[mask])

    if not relative:
        return raw_powers

    total = sum(raw_powers.values())
    if total == 0:
        return {b: 0.0 for b in raw_powers}
    return {b: p / total for b, p in raw_powers.items()}


def extract_features(epochs, sfreq=SFREQ, bands=BANDS, relative=True):
    """
    calculate band power for all epochs.

 input: epochs (n_epochs, n_samples)
 output: power values for each epoch and frequency band,
        along with the band names in column order.
    """
    band_names = list(bands.keys())
    feature_rows = []

    for epoch in epochs:
        powers = band_power(epoch, sfreq, bands, relative=relative)
        feature_rows.append([powers[b] for b in band_names])

    return np.array(feature_rows), band_names


def spectral_entropy(epoch, sfreq=SFREQ):
    """
    calculate shannon entropy of the normalized power spectrum.
    low entropy = power is concentrated in a few frequencies.
    high entropy = power is spread across many frequencies.
    useful for telling apart sleep stages with different frequency patterns.
    """
    freqs, psd = welch(epoch, fs=sfreq, nperseg=sfreq * 4)
    psd_norm = psd / (psd.sum() + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return entropy


def band_ratios(powers):
    """
   calculate common eeg power ratios.
    alpha/delta: usually lower in deep sleep and higher during wake/light sleep.
    theta/beta: often higher during drowsiness and light sleep.
    a small epsilon prevents division by zero in very quiet epochs.
    """
    eps = 1e-12
    return {
        "alpha_delta_ratio": powers["alpha"] / (powers["delta"] + eps),
        "theta_beta_ratio": powers["theta"] / (powers["beta"] + eps),
    }


def eog_features(eog_signal, sfreq=SFREQ):
    """
calculate simple time-based features for the eog channel.

variance: shows how much the signal changes. higher during rem.
max_abs: captures the largest eye movement or blink.
zero_crossing_rate: shows how often the signal changes direction,
 with higher values usually meaning faster movement.

    """
    variance = np.var(eog_signal)
    max_abs = np.max(np.abs(eog_signal))
    zero_crossings = np.sum(np.diff(np.sign(eog_signal)) != 0)
    zero_crossing_rate = zero_crossings / (len(eog_signal) / sfreq)
    return {"eog_variance": variance, "eog_max_abs": max_abs,
            "eog_zcr": zero_crossing_rate}


def extract_full_features(epochs, sfreq=SFREQ, bands=BANDS):
    """
    get the main features from the eeg and eog signals for each epoch.
    input: epochs (n_epochs, 2, n_samples)
    channel 0 = eeg, channel 1 = eog
    output: features for each epoch and their names.
    we get 10 features:
- eeg: delta, theta, alpha and beta power
- eeg: alpha/delta and theta/beta ratios
- eeg: spectral entropy
- eog: variance, max_abs and zero crossing rate
    """
    band_names = list(bands.keys())
    feature_names = (
        [f"eeg_{b}" for b in band_names] +
        ["alpha_delta_ratio", "theta_beta_ratio"] +
        ["spectral_entropy"] +
        ["eog_variance", "eog_max_abs", "eog_zcr"]
    )

    rows = []
    for epoch in epochs:
        eeg_signal, eog_signal = epoch[0], epoch[1]

        powers = band_power(eeg_signal, sfreq, bands, relative=True)
        ratios = band_ratios(powers)
        entropy = spectral_entropy(eeg_signal, sfreq)
        eog_feats = eog_features(eog_signal, sfreq)

        row = (
            [powers[b] for b in band_names] +
            [ratios["alpha_delta_ratio"], ratios["theta_beta_ratio"]] +
            [entropy] +
            [eog_feats["eog_variance"], eog_feats["eog_max_abs"], eog_feats["eog_zcr"]]
        )
        rows.append(row)

    return np.array(rows), feature_names


def add_temporal_context(features, subjects, window=1):
    """
    add information from nearby epochs to each epoch.
    this helps the model use the fact that sleep stages usually change
    gradually instead of randomly from one epoch to the next.
    for window=1, each epoch gets its previous, current and next features.
    neighbors are only taken from the same subject. for the first or last
    epoch, the current features are used instead of taking data from another
subject.
input:
    features: features for each epoch
    subjects: subject id for each epoch
    window: number of nearby epochs to include

output:
    features with the neighboring epochs added
    """
    n_epochs = features.shape[0]
    context_rows = []

    for i in range(n_epochs):
        parts = []
        for offset in range(-window, window + 1):
            j = i + offset
            if j < 0 or j >= n_epochs or subjects[j] != subjects[i]:
                j = i  # out of bounds or crosses subject boundary -> pad with self
            parts.append(features[j])
        context_rows.append(np.concatenate(parts))

    return np.array(context_rows)