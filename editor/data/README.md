# Frozen dataset layout

Place ImageNet source images under `images/` using each JSONL `source_id` verbatim:

```text
data/
  01-traces.jsonl
  images/
    00888/
      584755295296828.jpg
    00201/
      252218170729788.jpg
```

The ImageNet ground-truth index is already stored in each JSONL record as `ground_truth_label` and must remain 0-based. Do not create a second label mapping for this catalog.

Alternative locations can be selected at server start:

```bash
EDITTRACE_IMAGES=/absolute/path/to/images \
EDITTRACE_TRACES=/absolute/path/to/01-traces.jsonl \
EDITTRACE_MODEL=resnet50 \
python run.py
```

On Windows PowerShell, set the same names through `$env:EDITTRACE_IMAGES`, `$env:EDITTRACE_TRACES`, and `$env:EDITTRACE_MODEL` before running `py run.py`.
