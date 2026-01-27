"""
Static file serving and SPA routing.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import STATIC_DIR


def setup_static_files(app: FastAPI):
    """
    Configure static file serving for the SPA frontend.
    MUST be called AFTER all API routes are registered.
    """
    if not os.path.exists(STATIC_DIR):
        print(f"Static directory not found: {STATIC_DIR}")
        return

    # Mount assets directory
    assets_dir = os.path.join(STATIC_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend index.html for root path."""
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend not built. Please build frontend first."}

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Catch-all route for SPA routing.
        Returns static files if they exist, otherwise returns index.html.
        This allows client-side routing to work correctly.
        """
        # Check if it's a static file
        static_file = os.path.join(STATIC_DIR, full_path)
        if os.path.exists(static_file) and os.path.isfile(static_file):
            return FileResponse(static_file)

        # Otherwise return index.html (SPA routing)
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Not found")
