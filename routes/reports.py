"""
ARIA / Hermes — Reports API Routes
Lists generated weekly reports and triggers manual report generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])

REPORTS_DIR = Path(__file__).parent.parent / "reports"


@router.get("")
async def list_reports() -> list[dict[str, Any]]:
    """List all generated weekly reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
    return [
        {
            "filename": f.name,
            "path": str(f),
            "date": f.stem.replace("weekly_report_", ""),
            "size_kb": round(f.stat().st_size / 1024, 1),
        }
        for f in files
    ]


@router.get("/{filename}")
async def get_report(filename: str) -> FileResponse:
    """Serve a specific weekly report HTML file."""
    report_path = REPORTS_DIR / filename
    if not report_path.exists() or not report_path.suffix == ".html":
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="text/html")


@router.post("/generate")
async def trigger_report_generation() -> JSONResponse:
    """Manually trigger weekly report generation."""
    try:
        from services.report_service import generate_weekly_report
        path = await generate_weekly_report()
        return JSONResponse({
            "status": "ok",
            "message": "Report generated successfully",
            "path": path,
        })
    except Exception as e:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(e))
