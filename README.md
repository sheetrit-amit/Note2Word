# Note2Word

RNN-based lyric generation conditioned on melody (MIDI), with two melody-integration variants: a static summary vector and a time-bucketed sequence.

## Requirements & install
1) Activate the existing venv:  
   `source .venv/bin/activate`
2) Install deps (bypass CodeArtifact if configured):  
   `PIP_INDEX_URL=https://pypi.org/simple pip install -r requirements.txt`  
   Optional: `python -m pip install --upgrade pip`
3) Main deps: PyTorch, gensim, pandas, pretty-midi, tensorboard.

## Data (kept out of git)
- MIDI: `/home/amit/Downloads/Archive/midi_files/` (625 files)
- Lyrics CSV:
  - Train: `/home/amit/Downloads/Archive/lyrics_train_set.csv`
  - Test: `/home/amit/Downloads/Archive/lyrics_test_set.csv`

MIDI files are matched by normalized `artist + title`. Ensure these paths exist locally before running.

## Training
Static variant (one melody summary vector concatenated to every token):
```
python src/train.py --variant static \
  --train_csv /home/amit/Downloads/Archive/lyrics_train_set.csv \
  --midi_dir /home/amit/Downloads/Archive/midi_files \
  --epochs 2 --batch_size 8
```

Temporal variant (time-bucketed melody features concatenated per step):
```
python src/train.py --variant temporal \
  --train_csv /home/amit/Downloads/Archive/lyrics_train_set.csv \
  --midi_dir /home/amit/Downloads/Archive/midi_files \
  --bucket_seconds 2.0 --epochs 2 --batch_size 8
```

Artifacts:
- TensorBoard: `runs/<variant>/`
- Checkpoints: `checkpoints/<variant>_best.pt`

## Generation
After training, e.g., static variant:
```
python src/generate.py \
  --ckpt checkpoints/static_best.pt \
  --midi_dir /home/amit/Downloads/Archive/midi_files \
  --test_csv /home/amit/Downloads/Archive/lyrics_test_set.csv \
  --start_word hello --max_len 50 \
  --output generated_static.json
```
Temporal variant: swap to `temporal_best.pt`.

## Quick run guide (end-to-end)
1) Activate venv:  
   `source .venv/bin/activate`
2) Install deps (bypass CodeArtifact if needed):  
   `PIP_INDEX_URL=https://pypi.org/simple pip install -r requirements.txt`  
   Optional: `python -m pip install --upgrade pip`
3) Train static (baseline):  
   `python src/train.py --variant static --train_csv /home/amit/Downloads/Archive/lyrics_train_set.csv --midi_dir /home/amit/Downloads/Archive/midi_files --epochs 2 --batch_size 8`
   - Logs: `runs/static/`  
   - Checkpoint: `checkpoints/static_best.pt`
4) Train temporal:  
   `python src/train.py --variant temporal --train_csv /home/amit/Downloads/Archive/lyrics_train_set.csv --midi_dir /home/amit/Downloads/Archive/midi_files --bucket_seconds 2.0 --epochs 2 --batch_size 8`
   - Logs: `runs/temporal/`  
   - Checkpoint: `checkpoints/temporal_best.pt`
5) TensorBoard:  
   `tensorboard --logdir runs`
6) Generate (after training):  
   - Static:  
     `python src/generate.py --ckpt checkpoints/static_best.pt --midi_dir /home/amit/Downloads/Archive/midi_files --test_csv /home/amit/Downloads/Archive/lyrics_test_set.csv --start_word hello --max_len 50 --output generated_static.json`
   - Temporal:  
     `python src/generate.py --ckpt checkpoints/temporal_best.pt --midi_dir /home/amit/Downloads/Archive/midi_files --test_csv /home/amit/Downloads/Archive/lyrics_test_set.csv --start_word hello --max_len 50 --output generated_temporal.json`
7) Compare: check TensorBoard losses and generated JSONs for qualitative comparison.

## Code map
- `src/data.py`: CSV loading, tokenization, vocab, Word2Vec 300d training, MIDI feature extraction (static/temporal), Dataset + collate.
- `src/models.py`: Two GRU models — static (adds summary per step) and temporal (adds time-bucket feature per step).
- `src/train.py`: Train/val loop, TensorBoard logging, checkpointing.
- `src/generate.py`: Autoregressive generation with sampling (top-k/temperature), JSON output.

## Notes
- If pip points to CodeArtifact and returns 401, use `PIP_INDEX_URL=https://pypi.org/simple` or remove the index override from pip.conf/env.
- Data is not in git; place it at the paths above or adjust the CLI args accordingly.

