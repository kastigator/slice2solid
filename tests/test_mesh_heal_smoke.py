from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TestMeshHealSmoke(unittest.TestCase):
    def test_heal_produces_output_and_report(self) -> None:
        has_pymeshlab = importlib.util.find_spec("pymeshlab") is not None
        has_meshlabserver = shutil.which("meshlabserver") is not None or shutil.which("MeshLabServer") is not None
        if not (has_pymeshlab or has_meshlabserver):
            self.skipTest("No healing backend available (need pymeshlab or meshlabserver)")

        from slice2solid.mesh_heal import heal_mesh_file

        inp = ROOT / "data" / "samples" / "smoke_preview_structure.stl"
        self.assertTrue(inp.exists(), f"Missing sample mesh: {inp}")

        with tempfile.TemporaryDirectory(prefix="s2s_heal_test_") as td:
            out = Path(td) / "smoke_healed.stl"
            report = Path(td) / "heal_report.json"
            res = heal_mesh_file(inp, out_path=out, report_path=report, close_holes_max_mm=2.0)

            self.assertTrue(out.exists(), "Healed mesh file not created")
            self.assertTrue(report.exists(), "Report JSON not created")

            self.assertIsNotNone(res.before, "Missing pre-stats")
            self.assertIsNotNone(res.after, "Missing post-stats")
            self.assertGreater(res.after.vertices, 0)
            self.assertGreater(res.after.faces, 0)

            if res.before.boundary_edges is not None and res.after.boundary_edges is not None:
                self.assertLessEqual(
                    int(res.after.boundary_edges),
                    int(res.before.boundary_edges),
                    "Boundary edges should not increase for safe healing",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

