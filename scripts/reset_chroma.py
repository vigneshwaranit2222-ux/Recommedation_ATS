"""Explicitly reset only the local Chroma persistence directory."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb

from app.config import settings


def reset(confirm: bool) -> None:
    """Delete the configured local store only after explicit confirmation."""
    target = (ROOT / settings.CHROMA_PERSIST_DIR).resolve()
    if target.parent != ROOT.resolve() or target.name != "chroma_data":
        raise RuntimeError(f"Refusing to reset unsafe Chroma path: {target}")
    if not confirm:
        raise RuntimeError("Refusing destructive reset. Re-run with --confirm.")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target))
    metadata = {"hnsw:space": "cosine"}
    client.get_or_create_collection(settings.CHROMA_COLLECTION_JOBS, metadata=metadata)
    client.get_or_create_collection(settings.CHROMA_COLLECTION_RESUMES, metadata=metadata)
    print(f"Reset and initialized Chroma collections in {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset local Chroma collections.")
    parser.add_argument("--confirm", action="store_true", help="Confirm deletion of chroma_data.")
    reset(parser.parse_args().confirm)
