"""Ingest a folder of .txt / .md / .pdf files into the running service.

Usage:
    python scripts/load_folder.py [folder]      # defaults to sample_docs/

Env:
    RAG_API   base URL of the service (default http://localhost:8000)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_docs")
    api = os.environ.get("RAG_API", "http://localhost:8000")

    documents = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = read_file(path)
            if text.strip():
                documents.append({"source": str(path.relative_to(folder)), "text": text})

    if not documents:
        print(f"No supported documents found in {folder}/")
        return

    resp = httpx.post(f"{api}/ingest", json={"documents": documents}, timeout=300)
    resp.raise_for_status()
    print(f"Ingested {len(documents)} document(s):", resp.json())


if __name__ == "__main__":
    main()
