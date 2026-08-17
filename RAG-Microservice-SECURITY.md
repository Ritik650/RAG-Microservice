# Security Policy

## Scope

This service accepts file uploads (`/ingest/files`), calls an external LLM API (Gemini), and optionally gates access behind JWT bearer auth. Security reports are welcome, particularly for:

- **Auth** (`app/auth`) — JWT validation, the `AUTH_ENABLED`/`JWT_SECRET` handling, token expiry
- **File ingestion** (`app/ingest/parsers.py`) — parser handling of malformed/malicious PDF, DOCX, HTML, or CSV input
- **Secrets handling** — `.env` / `GEMINI_API_KEY` / `JWT_SECRET` exposure paths
- **Dependency vulnerabilities** in `requirements.txt`, `requirements-dev.txt`, `requirements-eval.txt`

## Supported Versions

This project does not maintain multiple release branches. Only the latest commit on `main` is supported.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| Older commits | ❌ |

## Known scope limitations

- `JWT_SECRET` and `GEMINI_API_KEY` are read from environment variables (`.env` locally, secrets in CI/deployment). Never commit a populated `.env` — `.env.example` should be the only version-controlled copy.
- Auth is **optional** (`AUTH_ENABLED=false` by default in local/dev setups). Don't deploy `/ingest` or `/query` publicly with auth disabled unless the corpus is intentionally public.
- Uploaded files are parsed but not sandboxed beyond what the underlying libraries (`pypdf`, `python-docx`) provide. Treat `/ingest/files` as trusted-input-only unless you've added your own sandboxing layer.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately via one of:
- GitHub's [private vulnerability reporting](https://github.com/Ritik650/RAG-Microservice/security/advisories/new) (Security tab → Report a vulnerability)
- Email: ry9812262@gmail.com

Please include a description of the vulnerability, steps to reproduce, and potential impact. This is a solo-maintained project, so response times aren't guaranteed, but reports will be acknowledged and addressed as soon as possible.
