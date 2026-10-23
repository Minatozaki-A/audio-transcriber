"""Transcript names, collision handling, and publication of completed drafts."""

import shutil
from datetime import date
from pathlib import Path

from utils.paths import get_output_directory


def _unique_path(base_path: Path, max_attempts: int = 1000) -> Path:
    """Find an unused path by adding a numeric suffix; does not reserve it."""
    if not base_path.exists():
        return base_path
    stem, suffix = base_path.stem, base_path.suffix
    for n in range(1, max_attempts + 1):
        candidate: Path = base_path.parent / f"{stem}-({n}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique path for {base_path} after {max_attempts} attempts")


def validate_transcript_name(name: str) -> None:
    if (not name or name.removesuffix(".md") in {"", ".", ".."}
            or "/" in name or "\\" in name):
        raise ValueError("Transcript name must be a file name, not a path")


def create_name_trans_path(name: str | None = None) -> Path:
    if name is None:
        new_name = f"trans-{date.today():%Y-%m-%d}.md"
    else:
        validate_transcript_name(name)
        new_name = f"{name.removesuffix('.md')}.md"

    final_name: Path = get_output_directory() / new_name

    return _unique_path(final_name)


def publish_transcript(draft_path: Path, output_path: Path) -> Path:
    """Copy a completed draft without overwriting existing transcripts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = _unique_path(output_path)
        try:
            destination = candidate.open("x", encoding="utf-8")
        except FileExistsError:
            continue
        try:
            with destination, draft_path.open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, destination)
        except BaseException as error:
            try:
                candidate.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"Cannot remove partial transcript {candidate}: {cleanup_error}")
            raise
        return candidate
