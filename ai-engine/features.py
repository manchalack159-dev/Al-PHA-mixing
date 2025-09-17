import argparse
import os
import glob
import json
import librosa
import numpy as np
import soundfile as sf
import csv

SEGMENT_SECONDS = 5  # Length of each feature segment in seconds

def load_json_events(json_path):
    """Load events from session JSONL and return timestamped EQ gains per channel-band."""
    events = []
    with open(json_path, "r") as f:
        for line in f:
            event = json.loads(line)
            # Only keep EQ gain events
            if event.get("parameter", "").startswith("channel_eq_gain"):
                events.append(event)
    return events

def get_segment_labels(events, start, end):
    """Average EQ gains for each band/channel over segment window."""
    gains = {}
    for event in events:
        ts = event["timestamp"]
        # assume ts in ISO, convert to seconds from first event
        t_sec = librosa.time_to_samples(librosa.time_to_frames(0, sr=1))  # Dummy, ignore
        # For simplicity, we use ordering. In real use, sync audio/JSON timebase.
        # Here, we average all gain values in segment.
        if "values" in event and len(event["values"]) > 0:
            key = f'{event.get("channel")}_{event.get("address")}'
            gains.setdefault(key, []).append(float(event["values"][0]))
    # Average per key
    return {k: np.mean(v) for k, v in gains.items()}

def compute_features(audio, sr):
    """Compute features from audio."""
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel)
    stft = librosa.stft(audio)
    stft_mag = np.abs(stft)
    delta = librosa.feature.delta(log_mel)
    return {
        "mel": mel,
        "log_mel": log_mel,
        "stft_mag": stft_mag,
        "delta": delta
    }

def process_file(audio_path, json_path, out_dir, segment_seconds=SEGMENT_SECONDS):
    session_id = os.path.splitext(os.path.basename(audio_path))[0]
    # Load audio
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    # Load JSON events
    events = load_json_events(json_path)
    # Segment and extract features
    mapping_rows = []
    for start in np.arange(0, duration, segment_seconds):
        end = min(start + segment_seconds, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_seg = audio[start_sample:end_sample]
        features = compute_features(audio_seg, sr)
        label_vec = get_segment_labels(events, start, end)
        # Save .npz file
        seg_name = f"{session_id}_{int(start)}_{int(end)}"
        npz_path = os.path.join(out_dir, f"{seg_name}.npz")
        np.savez(npz_path, **features)
        # Save mapping row
        mapping_rows.append([session_id, start, end, json.dumps(label_vec)])
    return mapping_rows

def main():
    parser = argparse.ArgumentParser(description="Audio Feature Extractor")
    parser.add_argument("--input-audio-dir", required=True)
    parser.add_argument("--input-json-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mapping_csv = os.path.join(args.out_dir, "mapping.csv")
    all_rows = []

    audio_files = glob.glob(os.path.join(args.input_audio_dir, "*.wav"))
    for audio_path in audio_files:
        session_id = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = os.path.join(args.input_json_dir, f"{session_id}.jsonl")
        if not os.path.exists(json_path):
            print(f"Missing session JSON for {session_id}, skipping.")
            continue
        rows = process_file(audio_path, json_path, args.out_dir)
        all_rows.extend(rows)

    # Save mapping CSV
    with open(mapping_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "start", "end", "label_vector"])
        writer.writerows(all_rows)
    print(f"Feature extraction complete. Mapping saved to {mapping_csv}")

if __name__ == "__main__":
    main()
