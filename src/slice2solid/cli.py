from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from slice2solid.mesh_heal import heal_mesh_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slice2solid", description="slice2solid utilities")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")

    sub = p.add_subparsers(dest="cmd", required=True)

    heal = sub.add_parser("heal", help="Heal an existing mesh file (STL/OBJ/PLY)")
    heal.add_argument("input", help="Input mesh path (.stl/.obj/.ply)")
    heal.add_argument("--heal", action="store_true", help="Compatibility flag (healing is enabled for this command)")
    heal.add_argument("--heal-preset", choices=["safe", "aggressive"], default="safe")
    heal.add_argument("--heal-out", default=None, help="Output path (default: рядом с исходным, *_healed)")
    heal.add_argument("--close-holes-max", type=float, default=2.0, help="Max hole size (mm), safe default: 2.0")
    heal.add_argument("--report", default=None, help="Write JSON report to this path")
    heal.add_argument(
        "--backend",
        choices=["auto", "pymeshlab", "meshlabserver"],
        default="auto",
        help="Healing backend preference",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(message)s")
    log = logging.getLogger("slice2solid.cli")

    if args.cmd == "heal":
        inp = Path(args.input)
        out = Path(args.heal_out) if args.heal_out else None
        report = Path(args.report) if args.report else None
        res = heal_mesh_file(
            inp,
            out_path=out,
            preset=str(args.heal_preset),
            close_holes_max_mm=float(args.close_holes_max),
            report_path=report,
            backend=str(args.backend),
        )
        log.info("Healed: %s", res.output_path)
        if report is None:
            summary = {
                "before": res.before.to_json() if res.before else None,
                "after": res.after.to_json() if res.after else None,
            }
            log.info("Summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
