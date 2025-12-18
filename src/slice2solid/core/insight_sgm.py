from __future__ import annotations

import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedSgm:
    source_path: Path
    extracted_path: Path


def extract_sgm_to_folder(job_dir: str | Path, out_dir: str | Path) -> ExtractedSgm | None:
    """
    Best-effort extraction of Insight SGM slice-geometry into a plain text file.

    Insight jobs commonly contain either:
    - `*.sgm.gz` (gzip compressed text), or
    - an extracted `*.sgm/part.sgm` directory layout produced by external tools.

    Some environments also pack it into a zip.
    """
    root = Path(job_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Standard: *.sgm.gz
    for gz_path in root.glob("*.sgm.gz"):
        try:
            raw = gzip.decompress(gz_path.read_bytes())
        except OSError:
            continue
        out_path = out_root / (gz_path.name[:-3])  # strip ".gz"
        out_path.write_bytes(raw)
        return ExtractedSgm(source_path=gz_path, extracted_path=out_path)

    # 2) Zip: any .zip containing a .sgm
    for zip_path in root.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                sgm_names = [n for n in zf.namelist() if n.lower().endswith(".sgm")]
                if not sgm_names:
                    continue
                # Prefer shortest name (often just "part.sgm").
                sgm_name = sorted(sgm_names, key=len)[0]
                raw = zf.read(sgm_name)
        except Exception:
            continue
        out_path = out_root / Path(sgm_name).name
        out_path.write_bytes(raw)
        return ExtractedSgm(source_path=zip_path, extracted_path=out_path)

    # 3) Directory layout: root/part.sgm/part.sgm
    for sgm_dir in root.glob("*.sgm"):
        if not sgm_dir.is_dir():
            continue
        inner = sgm_dir / sgm_dir.name
        if not inner.exists():
            # sometimes inner is just "part.sgm"
            candidates = list(sgm_dir.glob("*.sgm"))
            inner = candidates[0] if candidates else inner
        if inner.exists() and inner.is_file():
            out_path = out_root / inner.name
            out_path.write_bytes(inner.read_bytes())
            return ExtractedSgm(source_path=inner, extracted_path=out_path)

    return None

