import os
from huggingface_hub import snapshot_download
from pathlib import Path

DATASET_ID = "google/dreambooth"
DATA_DIR = os.environ.get("DATA_DIR", "data/google_dreambooth/dataset")
DEST_ROOT = Path(DATA_DIR)

snapshot_download(repo_id=DATASET_ID, repo_type="dataset", local_dir=DEST_ROOT)