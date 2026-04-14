"""Lightweight PLY point cloud viewer server."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/scenes")
async def list_scenes():
    if not OUTPUT_DIR.exists():
        return {"scenes": []}
    scenes = [
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.ply"))
    ]
    return {"scenes": sorted(scenes)}


@app.get("/ply/{scene}")
async def get_ply(scene: str):
    # Prevent path traversal
    scene_dir = OUTPUT_DIR / scene
    if not scene_dir.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid scene name")
    ply_files = list(scene_dir.glob("*.ply"))
    if not ply_files:
        raise HTTPException(status_code=404, detail=f"No PLY file found for scene '{scene}'")
    return FileResponse(ply_files[0], media_type="application/octet-stream")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7863))
    uvicorn.run(app, host="0.0.0.0", port=port)
