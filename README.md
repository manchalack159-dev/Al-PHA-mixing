# Genesis — Autonomous EQ Mixer

*"Let your mix flow through the machine—Genesis listens, learns, and brings harmony."*

## Quick Start

```bash
docker-compose up
```

## Data Collection

1. **Log X32 Mixer Sessions:**  
   Use `x32_logger` to record parameter changes and session events to JSONL files.
2. **Record Audio:**  
   Use the provided `recorder.py` to capture dry vocal tracks aligned by session ID, saving WAV files for each session.

## Training

1. **Extract Features:**  
   Run `features.py` to process audio and session logs into feature segments and labels.
2. **Build Dataset:**  
   Use `make_dataset.py` to assemble training/test splits, normalize features, and save scalers.
3. **Train Model:**  
   Execute `train.py` to train the CNN regression model on mel-spectrogram features.
4. **Evaluate Model:**  
   Use `evaluate.py` to compute metrics, visualize predictions, and export reports.

## Deployment

- **Serve the Model:**  
  Start `serve_model.py` (FastAPI) for real-time inference from audio or features.
- **Apply EQ:**  
  Use `apply_eq.py` to send predicted EQ gains to the X32 mixer via OSC, with safety smoothing and logging.

---

## License

See [LICENSE](LICENSE) (MIT).

---

## TODO

See [TODO.md](TODO.md) for planned features.