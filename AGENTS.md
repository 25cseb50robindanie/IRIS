# AGENTS.md

Offline desktop app for semantic search and change detection over satellite imagery (SIH Problem Statement 26227). Electron + React frontend, Python/FastAPI backend. See `architecture.md` for full system design and rationale — this file is operational only.

## Project structure

```
iris-project/
├── frontend/
│   ├── electron/          # main.js, preload.js
│   ├── src/
│   │   ├── components/    # React components (MapView, SearchBar, Sidebar, StatusBar)
│   │   ├── pages/         # App pages
│   │   └── styles/        # Tailwind config, global CSS
│   └── package.json
├── backend/
│   ├── main.py            # FastAPI entry + titiler mount
│   ├── api/               # Route handlers (ingestion, search, change)
│   ├── ingestion/         # Universal loader, capability detection, masking, COG conversion
│   ├── change_detection/  # Phases 1-5 of the change pipeline
│   ├── embedding/         # RemoteCLIP inference, FAISS index management
│   ├── catalog/           # SQLite init, schema, queries (database.py)
│   └── requirements.txt
├── skills/                # Agent skill files
├── data/
│   ├── uploads/           # Raw imported files (before processing)
│   ├── cogs/              # Converted COGs (production files)
│   └── staging/           # Temp outputs (atomic rename on success)
├── architecture.md
└── AGENTS.md              # This file
```

## Setup

```bash
# Frontend
cd frontend && npm install

# Backend
cd backend && pip install -r requirements.txt --break-system-packages
```

## Run

```bash
# Backend (terminal 1)
cd backend && uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend dev (terminal 2)
cd frontend && npm run dev

# Electron + frontend (terminal 2, alternative)
cd frontend && npm run electron:dev
```

## Code style

- Python: Black formatting, type hints required on new functions
- JS/TS: Prettier + ESLint
- No commented-out debug code in commits

## Hard rules

- **No network calls except to `localhost`.** Fully offline.
- **No login/auth system.** Single-analyst local tool.
- **Respect module boundaries.** `ingestion/`, `embedding/`, `change_detection/`, `catalog/`, `api/`, `frontend/src/components/` are separate concerns.
- **Follow `skills/secure-coding-standards/SKILL.md`** for file I/O, the local server, or Electron config.
- **Electron:** `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` — do not override.
- **SQLite:** WAL mode, connection-per-process after fork, `BEGIN IMMEDIATE` for write transactions. See `backend/catalog/database.py`.
- **Backend binds to `127.0.0.1` only.** Never `0.0.0.0`.

## Build order (implement in this sequence)

1. **Milestone 1 — See imagery on screen**
   - Backend: universal loader → COG conversion → titiler serves tiles
   - Frontend: Electron shell + MapLibre rendering COG tiles from localhost
   - Catalog: SQLite init, scene record on import

2. **Milestone 2 — Semantic search works**
   - Backend: tiling → RemoteCLIP embedding → FAISS index
   - Frontend: search bar → results list → fly-to-location

3. **Milestone 3 — Change detection pipeline**
   - Backend: Phases 1-5 as a background job
   - Frontend: before/after comparison view, change highlights

4. **Milestone 4 — Review queue and export**
   - Frontend: confirm/reject UI, audit trail
   - Backend: GeoJSON export with provenance

## Where to look

- `architecture.md` — full pipeline design, rationale, known limitations, decision log
- `skills/secure-coding-standards/SKILL.md` — security requirements
