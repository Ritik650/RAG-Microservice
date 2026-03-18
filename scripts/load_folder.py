"""Ingest a folder of documents into the running service.

Supports every format the service parses: .pdf, .docx, .html/.htm, .csv/.tsv,
.txt, .md, .rst, .log (see app/ingest/parsers.py — one registry for both paths).

Usage:
    python scripts/load_folder.py [folder]      # defaults to sample_docs/

Env:
    RAG_API   base URL of the service (default http://localhost:8000)
    RAG_TOKEN bearer token, only needed when AUTH_ENABLED=true
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.parsers import SUPPORTED_EXTENSIONS, parse_bytes  # noqa: E402


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_docs")
    api = os.environ.get("RAG_API", "http://localhost:8000")
    headers = {}
    if token := os.environ.get("RAG_TOKEN"):
        headers["authorization"] = f"Bearer {token}"

    documents = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                text = parse_bytes(path.name, path.read_bytes())
            except Exception as exc:  # keep going; report the bad file
                print(f"  ! skipping {path}: {exc}")
                continue
            if text.strip():
                documents.append({"source": str(path.relative_to(folder)), "text": text})

    if not documents:
        print(f"No supported documents found in {folder}/")
        return

    resp = httpx.post(
        f"{api}/ingest", json={"documents": documents}, headers=headers, timeout=300
    )
    resp.raise_for_status()
    print(f"Ingested {len(documents)} document(s):", resp.json())


if __name__ == "__main__":
    main()
