from pathlib import Path


def scan_directory(directory: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    reference_files = tuple(directory.glob("ref_*"))
    hypothesis_files = tuple(directory.glob("hyp_*"))
    return reference_files, hypothesis_files
