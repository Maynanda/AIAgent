"""
ARIA / Hermes — Filesystem Tools
Gives Hermes sandboxed access to read, write, search, and reason about
local files and folders on the machine.

Security model:
  - Only paths under FILESYSTEM_ALLOWED_PATHS are accessible
  - Blocked file extensions (.env, .key, .pem, etc.) cannot be read
  - No symlink traversal outside allowed roots
  - Files over FILESYSTEM_MAX_FILE_BYTES are summarized, not fully read
  - Vision: image files are described by Qwen instead of dumped as bytes
"""
from __future__ import annotations

import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ── Security helpers ──────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".log",
    ".py", ".js", ".ts", ".html", ".css", ".sh", ".sql",
    ".xml", ".env.example",
}


def _get_allowed_roots() -> list[Path]:
    """Parse the FILESYSTEM_ALLOWED_PATHS setting into Path objects."""
    raw = settings.filesystem_allowed_paths.strip()
    if not raw:
        return []
    return [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]


def _get_blocked_extensions() -> set[str]:
    raw = settings.filesystem_blocked_extensions.strip()
    return {ext.strip().lower() for ext in raw.split(",") if ext.strip()}


def _is_path_allowed(path: Path) -> tuple[bool, str]:
    """Return (is_allowed, reason). Resolves symlinks before checking."""
    try:
        resolved = path.resolve()
    except Exception as e:
        return False, f"Cannot resolve path: {e}"

    allowed_roots = _get_allowed_roots()
    if not allowed_roots:
        return False, "No FILESYSTEM_ALLOWED_PATHS configured. Add paths to .env to enable file access."

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return True, ""
        except ValueError:
            continue

    return False, f"Path '{resolved}' is outside all allowed directories: {[str(r) for r in allowed_roots]}"


def _is_extension_blocked(path: Path) -> bool:
    return path.suffix.lower() in _get_blocked_extensions()


# ── Tools ─────────────────────────────────────────────────────────────────────

async def list_directory(path: str, show_hidden: bool = False) -> str:
    """
    List files and subdirectories in a folder.
    Returns a structured text listing with file sizes and modification times.
    """
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists():
        return f"❌ Path does not exist: {path}"
    if not target.is_dir():
        return f"❌ Not a directory: {path}"

    entries = []
    try:
        for entry in sorted(target.iterdir()):
            if not show_hidden and entry.name.startswith("."):
                continue
            if entry.is_dir():
                child_count = sum(1 for _ in entry.iterdir()) if entry.is_dir() else 0
                entries.append(f"📁  {entry.name}/  ({child_count} items)")
            elif entry.is_file():
                size_kb = entry.stat().st_size / 1024
                mod_time = datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                entries.append(f"📄  {entry.name}  ({size_kb:.1f} KB, modified {mod_time})")
    except PermissionError:
        return f"❌ Permission denied reading directory: {path}"

    if not entries:
        return f"📂 Directory is empty: {path}"

    return f"📂 Contents of {path}:\n" + "\n".join(entries)


async def read_file(path: str) -> str:
    """
    Read the content of a text file.
    For image files, returns a Qwen vision description instead of raw bytes.
    Files over the size limit are truncated with a summary note.
    """
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists():
        return f"❌ File not found: {path}"
    if not target.is_file():
        return f"❌ Not a file: {path}"
    if _is_extension_blocked(target):
        return f"❌ Blocked file type: {target.suffix} files cannot be read for security reasons."

    ext = target.suffix.lower()

    # Handle images via vision model
    if ext in _IMAGE_EXTENSIONS:
        return await _describe_image_file(target)

    # Handle binary files (non-text, non-image)
    if ext not in _TEXT_EXTENSIONS:
        mime, _ = mimetypes.guess_type(str(target))
        if mime and not mime.startswith("text"):
            return f"⚠️ Binary file ({mime}). Cannot display content of {target.name}. File size: {target.stat().st_size / 1024:.1f} KB"

    # Read text file
    try:
        file_size = target.stat().st_size
        max_bytes = settings.filesystem_max_file_bytes

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            if file_size > max_bytes:
                content = f.read(max_bytes)
                return (
                    f"📄 {path} (showing first {max_bytes // 1024}KB of {file_size // 1024}KB):\n\n"
                    f"{content}\n\n[...truncated — file too large to read fully]"
                )
            content = f.read()

        return f"📄 {path}:\n\n{content}"
    except Exception as e:
        return f"❌ Error reading file: {e}"


async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Write content to a file. Creates parent directories if needed.
    By default, refuses to overwrite existing files unless overwrite=True.
    """
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if _is_extension_blocked(target):
        return f"❌ Cannot write blocked file type: {target.suffix}"

    if target.exists() and not overwrite:
        return f"❌ File already exists: {path}. Set overwrite=True to replace it."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"✅ Written {len(content)} characters to {path}"
    except Exception as e:
        return f"❌ Error writing file: {e}"


async def search_files(directory: str, pattern: str, max_results: int = 30) -> str:
    """
    Recursively search for files matching a glob pattern inside a directory.
    Example: search_files('/Users/alpha/Documents', '*.pdf')
    """
    target = Path(directory).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists() or not target.is_dir():
        return f"❌ Directory not found: {directory}"

    try:
        matches = list(target.rglob(pattern))[:max_results]
        if not matches:
            return f"🔍 No files matching '{pattern}' found in {directory}"

        lines = [f"🔍 Found {len(matches)} file(s) matching '{pattern}' in {directory}:"]
        for m in matches:
            size_kb = m.stat().st_size / 1024
            lines.append(f"  {m.relative_to(target)}  ({size_kb:.1f} KB)")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Search failed: {e}"


async def get_file_info(path: str) -> str:
    """Get metadata about a file: size, type, created/modified timestamps."""
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists():
        return f"❌ Not found: {path}"

    stat = target.stat()
    mime, _ = mimetypes.guess_type(str(target))
    return (
        f"📋 File info: {target.name}\n"
        f"  Path:      {target.resolve()}\n"
        f"  Type:      {'Directory' if target.is_dir() else (mime or target.suffix or 'Unknown')}\n"
        f"  Size:      {stat.st_size / 1024:.1f} KB\n"
        f"  Modified:  {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Created:   {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}"
    )


async def create_directory(path: str) -> str:
    """Create a directory (and all parent directories)."""
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    try:
        target.mkdir(parents=True, exist_ok=True)
        return f"✅ Directory created: {path}"
    except Exception as e:
        return f"❌ Error creating directory: {e}"


async def move_file(source: str, destination: str) -> str:
    """Move or rename a file or folder within allowed paths."""
    src = Path(source).expanduser()
    dst = Path(destination).expanduser()

    src_ok, src_reason = _is_path_allowed(src)
    dst_ok, dst_reason = _is_path_allowed(dst)

    if not src_ok:
        return f"❌ Source access denied: {src_reason}"
    if not dst_ok:
        return f"❌ Destination access denied: {dst_reason}"

    if not src.exists():
        return f"❌ Source not found: {source}"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"✅ Moved: {source} → {destination}"
    except Exception as e:
        return f"❌ Move failed: {e}"


async def delete_file(path: str) -> str:
    """
    Delete a file (NOT a directory). Requires explicit allowed path check.
    Directories must be deleted via the shell or explicit tool — this is intentional.
    """
    target = Path(path).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists():
        return f"❌ Not found: {path}"
    if target.is_dir():
        return f"❌ '{path}' is a directory. Cannot delete directories with this tool (safety measure)."

    try:
        target.unlink()
        return f"✅ Deleted: {path}"
    except Exception as e:
        return f"❌ Delete failed: {e}"


async def index_folder_to_knowledge_graph(directory: str, db=None) -> str:
    """
    Recursively index text files in a folder into the knowledge graph.
    Extracts entities and relations from each file and stores them in the DB.
    """
    target = Path(directory).expanduser()
    allowed, reason = _is_path_allowed(target)
    if not allowed:
        return f"❌ Access denied: {reason}"

    if not target.exists() or not target.is_dir():
        return f"❌ Directory not found: {directory}"

    if db is None:
        from database.connection import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            return await _do_index(target, db)
    return await _do_index(target, db)


async def _do_index(target: Path, db) -> str:
    """Internal indexing logic."""
    from knowledge_graph.builder import KnowledgeGraphBuilder
    builder = KnowledgeGraphBuilder(db)

    indexed = 0
    skipped = 0
    errors = 0

    for file_path in target.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _TEXT_EXTENSIONS:
            skipped += 1
            continue
        if _is_extension_blocked(file_path):
            skipped += 1
            continue
        if file_path.stat().st_size > settings.filesystem_max_file_bytes:
            skipped += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            source_ref = f"file:{file_path.resolve()}"
            await builder.process_text(
                text=content[:3000],  # cap at 3k chars per file for indexing
                source_type="file",
                source_id=source_ref,
            )
            indexed += 1
        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")
            errors += 1

    return (
        f"✅ Folder indexing complete for {target}:\n"
        f"  Indexed: {indexed} files\n"
        f"  Skipped: {skipped} files (binary/blocked/too large)\n"
        f"  Errors:  {errors}"
    )


# ── Vision helper ─────────────────────────────────────────────────────────────

async def _describe_image_file(path: Path) -> str:
    """Use Qwen vision to describe an image file."""
    try:
        image_bytes = path.read_bytes()
        from llm.client import llm
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Describe the contents of the provided image clearly and concisely."},
            {"role": "user", "content": "Please describe what you see in this image."},
        ]
        description = await llm.generate(messages, images=[image_bytes], max_new_tokens=300)
        return f"🖼️ Image: {path.name}\n\n{description}"
    except Exception as e:
        return f"🖼️ Image file: {path.name} ({path.stat().st_size / 1024:.1f} KB) — vision description failed: {e}"
