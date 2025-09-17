import argparse
import os
import numpy as np
import csv
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_mapping(mapping_csv):
    """Load mapping CSV, return list of (npz_path, label_dict, start, end, session_id)."""
    mapping = []
    with open(mapping_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session_id = row["session_id"]
            start = float(row["start"])
            end = float(row["end"])
            label_vector = eval(row["label_vector"])
            npz_name = f"{session_id}_{int(start)}_{int(end)}.npz"
            mapping.append((npz_name, label_vector, start, end, session_id))
    return mapping

def window_features(feat, window_size, overlap):
    """Slice features into windows (axis=1 is time)."""
    # Assume feat shape: (feature_dim, time_frames)
    feat_dim, total_frames = feat.shape
    step = window_size - overlap
    windows = []
    for start in range(0, total_frames - window_size + 1, step):
        window = feat[:, start:start + window_size]
        windows.append(window)
    return windows

def get_label_vector(label_dict, label_keys):
    """Get label vector in consistent order."""
    return np.array([label_dict.get(k, 0.0) for k in label_keys], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Make ML dataset from features and mapping")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--window-size", type=int, default=50, help="Window size (frames)")
    parser.add_argument("--overlap", type=int, default=0, help="Window overlap (frames)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mapping = load_mapping(args.mapping_csv)
    # Collect all label keys for consistent ordering
    label_keys = set()
    for _, label_dict, *_ in mapping:
        label_keys.update(label_dict.keys())
    label_keys = sorted(label_keys)

    X = []
    Y = []
    npz_files = {os.path.basename(f): f for f in glob.glob(os.path.join(args.feature_dir, "*.npz"))}

    for npz_name, label_dict, *_ in mapping:
        npz_path = npz_files.get(npz_name)
        if npz_path is None:
            print(f"Missing feature file {npz_name}, skipping.")
            continue
        data = np.load(npz_path)
        # Use log_mel and delta as features (can adjust as needed)
        log_mel = data["log_mel"]  # shape: (mel_bins, frames)
        delta = data["delta"]      # shape: (mel_bins, frames)
        feat = np.concatenate([log_mel, delta], axis=0)  # shape: (2*mel_bins, frames)
        windows = window_features(feat, args.window_size, args.overlap)
        label_vec = get_label_vector(label_dict, label_keys)
        for w in windows:
            X.append(w.flatten())
            Y.append(label_vec)

    X = np.array(X)
    Y = np.array(Y)
    print(f"Dataset shapes: X={{X.shape}}, Y={{Y.shape}}")

    # Normalize features and labels
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    Y_scaled = Y_scaler.fit_transform(Y)

    joblib.dump(X_scaler, os.path.join(args.out_dir, "X_scaler.joblib"))
    joblib.dump(Y_scaler, os.path.join(args.out_dir, "Y_scaler.joblib"))

    # Train/test split (deterministic)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=args.test_size, random_state=args.random_state
    )
    np.savez(os.path.join(args.out_dir, "train.npz"), X=X_train, Y=Y_train)
    np.savez(os.path.join(args.out_dir, "test.npz"), X=X_test, Y=Y_test)
    print(f"Saved train/test splits and scalers to {{args.out_dir}}")

if __name__ == "__main__":
    main()
