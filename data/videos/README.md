
# Data Directory

Place your generated videos and prompt files here.

## Single Evaluation

Videos must be named `sample_0000.mp4`, `sample_0001.mp4`, ...  
Provide `prompts_shard_*.txt` files (format: `id<TAB>text`).  
Run `python scripts/prepare_data.py --video_dir data/videos/your_model_name` to extract audio and generate the metadata CSV.

## Batch Evaluation (variance)

Organize your data by prompt and seed:

```
data/videos/your_model/
├── prompts.json
├── prompt_001/
│   ├── seed_00.mp4
│   ├── seed_01.mp4
│   └── seed_02.mp4
└── ...
```

`prompts.json` format:
```json
[
    {"id": "prompt_001", "text": "A cat playing the piano."},
    {"id": "prompt_002", "text": "Waves crashing on a rocky shore."}
]
```

Then run `python scripts/run_batch_eval.py --model your_model --n_seeds 3`.
