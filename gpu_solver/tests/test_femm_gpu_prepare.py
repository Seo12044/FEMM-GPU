import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "femm_gpu_prepare.py"
SPEC = importlib.util.spec_from_file_location("femm_gpu_prepare", MODULE_PATH)
femm_gpu_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(femm_gpu_prepare)


class FemmGpuPrepareTests(unittest.TestCase):
    def test_builds_neutral_artifact_from_triangle_mesh_without_mutating_source(self):
        fixture = Path(__file__).parent / "fixtures" / "linear_square_v1" / "linear_square.fem"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fem"
            source.write_bytes(fixture.read_bytes())
            before = source.read_bytes()
            stem = root / "mesh"
            stem.with_suffix(".node").write_text(
                "4 2 0 1\n0 0 0 0\n1 1 0 0\n2 1 1 0\n3 0 1 0\n", encoding="ascii")
            stem.with_suffix(".ele").write_text(
                "2 3 1\n1 0 2 3 1\n0 0 1 2 1\n", encoding="ascii")
            stem.with_suffix(".edge").write_text(
                "5 1\n2 2 3 -2\n0 0 1 -2\n4 0 2 0\n3 3 0 -2\n1 1 2 -2\n", encoding="ascii")
            stem.with_suffix(".pbc").write_text("0\n", encoding="ascii")
            resolved = femm_gpu_prepare._build_resolved(source, stem)
            artifact = root / "artifact.json"
            femm_gpu_prepare._write_artifact(artifact, resolved, False)
            document = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(before, source.read_bytes())

        self.assertEqual(document["schema_version"], "gpu_femm_planar_dc_mesh_v1")
        self.assertNotIn("base_motor_fem_sha256", document)
        self.assertNotIn("pose", document["resolved"])
        self.assertEqual(document["source_fem_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(document["resolved"]["nodes_mm"][2], [1000.0, 1000.0])
        self.assertEqual(document["resolved"]["outer_dirichlet"]["node_indices"], [0, 1, 2, 3])
        self.assertEqual(document["resolved"]["triangles"]["region_ids"], [0, 0])
        self.assertEqual(document["resolved"]["triangles"]["node_indices"], [[0, 1, 2], [0, 2, 3]])

    def test_rejects_duplicate_element_ids(self):
        fixture = Path(__file__).parent / "fixtures" / "linear_square_v1" / "linear_square.fem"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fem"
            source.write_bytes(fixture.read_bytes())
            stem = root / "mesh"
            stem.with_suffix(".node").write_text(
                "4 2 0 1\n0 0 0 0\n1 1 0 0\n2 1 1 0\n3 0 1 0\n", encoding="ascii")
            stem.with_suffix(".ele").write_text(
                "2 3 1\n0 0 1 2 1\n0 0 2 3 1\n", encoding="ascii")
            stem.with_suffix(".edge").write_text(
                "4 1\n0 0 1 -2\n1 1 2 -2\n2 2 3 -2\n3 3 0 -2\n", encoding="ascii")
            stem.with_suffix(".pbc").write_text("0\n", encoding="ascii")
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "element row"):
                femm_gpu_prepare._build_resolved(source, stem)

    def test_existing_artifact_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.json"
            artifact.write_text("keep me\n", encoding="utf-8")
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "already exists"):
                femm_gpu_prepare._write_artifact(
                    artifact, {"source_fem_sha256": "0" * 64}, False
                )
            self.assertEqual(artifact.read_text(encoding="utf-8"), "keep me\n")

    def test_prepare_refuses_source_and_stock_femm_output_paths(self):
        fixture = Path(__file__).parent / "fixtures" / "linear_square_v1" / "linear_square.fem"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fem"
            source.write_bytes(fixture.read_bytes())
            stock = root / "stock_femm"
            stock.mkdir()
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "input FEMM model"):
                femm_gpu_prepare.prepare(source, source, stock, root / "noop.exe", 1, True)
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "stock FEMM"):
                femm_gpu_prepare.prepare(
                    source, stock / "artifact.json", stock, root / "noop.exe", 1, True
                )
            self.assertEqual(source.read_bytes(), fixture.read_bytes())

    def test_rejects_boundary_marker_on_interior_edge(self):
        fixture = Path(__file__).parent / "fixtures" / "linear_square_v1" / "linear_square.fem"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fem"
            source.write_bytes(fixture.read_bytes())
            stem = root / "mesh"
            stem.with_suffix(".node").write_text(
                "4 2 0 1\n0 0 0 0\n1 1 0 0\n2 1 1 0\n3 0 1 0\n", encoding="ascii")
            stem.with_suffix(".ele").write_text(
                "2 3 1\n0 0 1 2 1\n1 0 2 3 1\n", encoding="ascii")
            stem.with_suffix(".edge").write_text(
                "5 1\n0 0 1 -2\n1 1 2 -2\n2 2 3 -2\n3 3 0 -2\n4 0 2 -2\n",
                encoding="ascii",
            )
            stem.with_suffix(".pbc").write_text("0\n", encoding="ascii")
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "interior edge"):
                femm_gpu_prepare._build_resolved(source, stem)

    def test_rejects_periodic_triangle_mesh(self):
        fixture = Path(__file__).parent / "fixtures" / "linear_square_v1" / "linear_square.fem"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.fem"
            source.write_bytes(fixture.read_bytes())
            stem = root / "mesh"
            stem.with_suffix(".node").write_text("3 2 0 1\n0 0 0 0\n1 1 0 0\n2 0 1 0\n", encoding="ascii")
            stem.with_suffix(".ele").write_text("1 3 1\n0 0 1 2 1\n", encoding="ascii")
            stem.with_suffix(".edge").write_text("3 1\n0 0 1 -2\n1 1 2 -2\n2 2 0 -2\n", encoding="ascii")
            stem.with_suffix(".pbc").write_text("1\n", encoding="ascii")
            with self.assertRaisesRegex(femm_gpu_prepare.PrepareError, "periodic"):
                femm_gpu_prepare._build_resolved(source, stem)


if __name__ == "__main__":
    unittest.main()
