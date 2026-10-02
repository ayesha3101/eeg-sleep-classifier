"""
download the sleep-edf expanded recordings using mnes built in fetcher
"""
import argparse
from pathlib import Path
import mne
from mne.datasets.sleep_physionet.age import fetch_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(n_subjects: int = 10):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mne.set_config("MNE_DATASETS_SLEEPPHYSIONET_PATH", str(DATA_DIR))

    subjects = list(range(n_subjects))
    paths = fetch_data(subjects=subjects, recording=[1])

    print(f"Downloaded {len(paths)} subject recording pairs to {DATA_DIR}")
    for psg_path, hyp_path in paths:
        print(f"  PSG: {psg_path}")
        print(f"  Hypnogram: {hyp_path}")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=10,
                         help="Number of subjects to download (each 1 night of EEG)")
    args = parser.parse_args()
    download(args.n_subjects)