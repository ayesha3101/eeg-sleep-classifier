"""
batch process all sleep edf psg and hypnogram files in 
data/raw/ and save them as labeled epoch arrays in data/processed/
"""

import re
from pathlib import Path
import numpy as np
from preprocess import load_and_epoch_real_recording, trim_wake_periods

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def find_subject_pairs(raw_dir):
    """
    find matching psg and hypnogram files
    the first 6 characters of the filename identify the same subject and night
    example: sc4001e0 psg and sc4001ec hypnogram belong to the same subject and night
    """
    
    psg_files = sorted(raw_dir.rglob("*PSG.edf"))
    hyp_files = sorted(raw_dir.rglob("*Hypnogram.edf"))

    hyp_lookup = {f.name[:6]: f for f in hyp_files}

    pairs = []
    for psg in psg_files:
        key = psg.name[:6]
        if key in hyp_lookup:
            pairs.append((psg, hyp_lookup[key]))
        else:
            print(f"no matching hypnogram found for {psg.name}, skipping")

    return pairs


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pairs = find_subject_pairs(RAW_DIR)

    if not pairs:
        print(f"No PSG/Hypnogram pairs found under {RAW_DIR}")
        print("Make sure your downloaded .edf files are copied into data/raw/")
        return

    print(f"Found {len(pairs)} subject recording(s) to process\n")

    total_epochs = 0
    for psg_path, hyp_path in pairs:
        subject_id = psg_path.name[:6]
        print(f"Processing {subject_id} ...")

        try:
            epochs, labels = load_and_epoch_real_recording(str(psg_path), str(hyp_path))
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        n_before = len(labels)
        epochs, labels = trim_wake_periods(epochs, labels, buffer_epochs=60)
        n_after = len(labels)

        out_path = PROCESSED_DIR / f"{subject_id}.npz"
        np.savez(out_path, epochs=epochs, labels=labels)

        unique, counts = np.unique(labels, return_counts=True)
        stage_summary = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(f"  {n_before} epochs -> trimmed to {n_after} "
              f"(removed {n_before - n_after} excess Wake epochs)")
        print(f"  Saved to {out_path.name}")
        print(f"  Stage distribution: {stage_summary}\n")

        total_epochs += epochs.shape[0]

    print(f"Done. {total_epochs} total epochs across {len(pairs)} subjects "
          f"saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()