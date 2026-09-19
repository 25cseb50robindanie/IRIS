"""IRIS Backend — FastAPI + titiler, localhost only."""

import os
import re
import sys
import urllib.parse
from pathlib import Path

# On Windows with QGIS/GDAL, ensure DLL directories and PROJ paths are registered
if sys.platform == "win32":
    qgis_bin = Path(r"C:\Program Files\QGIS 3.44.14\bin")
    if qgis_bin.exists():
        try:
            os.add_dll_directory(str(qgis_bin))
        except Exception:
            pass
    if "PATH" in os.environ and str(qgis_bin) not in os.environ["PATH"]:
        os.environ["PATH"] = f"{qgis_bin};" + os.environ["PATH"]

if "PROJ_DATA" not in os.environ and "PROJ_LIB" not in os.environ:
    candidates = [
        Path(r"C:\Program Files\QGIS 3.44.14\share\proj"),
        Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "rasterio" / "proj_data",
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(candidate)
            os.environ["PROJ_LIB"] = str(candidate)
            try:
                import pyproj.datadir
                pyproj.datadir.set_data_dir(str(candidate))
            except Exception:
                pass
            break

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from titiler.application.main import app as titiler_app

from api.ingest import router as ingest_router

app = FastAPI(title="IRIS Backend", version="0.1.0")

# Allow the Electron frontend (localhost) to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def normalize_file_uri_middleware(request: Request, call_next):
    """Normalize file:/// URIs, drive letters, and percent-encoding for GDAL / rasterio."""
    if "url=" in request.url.query:
        query_dict = dict(urllib.parse.parse_qsl(request.url.query))
        if "url" in query_dict:
            raw_url = query_dict["url"]
            # 1. Decode percent-encoding (e.g., %27 -> ', %20 -> space)
            unquoted = urllib.parse.unquote(raw_url)
            # 2. Normalize Windows drive letter paths so GDAL sees the exact filesystem path
            if sys.platform == "win32":
                if re.match(r"^file:///[a-zA-Z]:", unquoted):
                    normalized_url = unquoted[8:]
                elif re.match(r"^file://[a-zA-Z]:", unquoted):
                    normalized_url = unquoted[7:]
                elif re.match(r"^/[a-zA-Z]:", unquoted):
                    normalized_url = unquoted[1:]
                else:
                    normalized_url = unquoted
            else:
                normalized_url = unquoted
            query_dict["url"] = normalized_url
            new_query = urllib.parse.urlencode(query_dict)
            request.scope["query_string"] = new_query.encode("latin-1")
    response = await call_next(request)
    if request.url.path.endswith("tilejson.json"):
        # titiler defaults to max-age=3600; a re-ingested scene must not serve a stale TileJSON
        response.headers["Cache-Control"] = "no-cache"
    return response


# Alias route for direct /tiles/cog/tilejson.json -> /tiles/cog/WebMercatorQuad/tilejson.json
@app.get("/tiles/cog/tilejson.json")
async def tilejson_alias(request: Request):
    query_str = request.url.query
    target = (
        f"/tiles/cog/WebMercatorQuad/tilejson.json?{query_str}"
        if query_str
        else "/tiles/cog/WebMercatorQuad/tilejson.json"
    )
    return RedirectResponse(url=target)


class _PrefixedApp:
    """Expose the mount prefix to a mounted ASGI app.

    Starlette builds ``request.base_url`` from ``app_root_path``, which a mounted
    sub-app does not inherit, so titiler's TileJSON would advertise tile URLs
    without ``/tiles`` (-> 404 for every tile).
    """

    def __init__(self, asgi_app, prefix: str) -> None:
        self.asgi_app = asgi_app
        self.prefix = prefix

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, "app_root_path": self.prefix}
        await self.asgi_app(scope, receive, send)


# Mount titiler at /tiles — serves COGs as XYZ map tiles on demand
app.mount("/tiles", _PrefixedApp(titiler_app, "/tiles"))

# Ingestion endpoint
app.include_router(ingest_router)


@app.get("/health")
def health():
    return {"status": "ok"}
