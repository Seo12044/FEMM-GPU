"""Keep the solver split into compiled modules, not textual source fragments."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NATIVE_TESTS = ROOT / "tests" / "native"
SOURCE_SUFFIXES = {".cpp", ".cu", ".h", ".cuh"}
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
COMMENTS_OR_QUOTES = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
GENERATED_AGE = "gpu_femm_native_age_matrix.inc"


def source_text(path):
    text = path.read_text(encoding="utf-8-sig")
    return COMMENTS_OR_QUOTES.sub(
        lambda match: "\n" * match[0].count("\n")
        if match[0].startswith(("//", "/*")) else match[0],
        text,
    )


def source_files(directory):
    return sorted(path for path in directory.rglob("*")
                  if path.suffix in SOURCE_SUFFIXES)


def include_target(path, name):
    for directory in (path.parent, SRC, NATIVE_TESTS):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


class SourceLayoutTests(unittest.TestCase):
    def test_entry_only_dispatches_to_cli(self):
        text = source_text(SRC / "linear_p1_poc.cu")
        self.assertEqual(INCLUDE.findall(text), ["app/cli.h"])
        body = INCLUDE.sub("", text).strip()
        self.assertRegex(
            body,
            r"\Aint\s+main\s*\(\s*int\s+argc\s*,\s*char\s*\*\s*\*\s*argv\s*\)"
            r"\s*\{\s*return\s+gpu_femm::RunCli\s*\(\s*argc\s*,\s*argv\s*\)\s*;\s*\}\s*\Z",
        )

    def test_sources_are_not_textually_included(self):
        for path in source_files(SRC) + source_files(NATIVE_TESTS):
            for name in INCLUDE.findall(source_text(path)):
                with self.subTest(file=path.relative_to(ROOT), include=name):
                    self.assertNotIn(Path(name).suffix.lower(), {".cpp", ".cu"})
                    if Path(name).suffix.lower() == ".inc":
                        self.assertEqual(name, GENERATED_AGE)
                        self.assertEqual(path, SRC / "native_airgap_matrix.cpp")

    def test_production_modules_do_not_contain_test_fixtures(self):
        definition = re.compile(
            r"^\s*(?:int|bool|Model|NonlinearModel)\s+"
            r"(?:SelfTest|\w+Tests|\w+ReferenceTest|\w+Fixture)"
            r"\s*\([^;{}]*\)\s*\{",
            re.MULTILINE,
        )
        for path in source_files(SRC):
            if path == SRC / "bh_curve_poc.cu":
                continue  # Separate B-H fixture executable, not the solver core.
            text = source_text(path)
            with self.subTest(file=path.relative_to(ROOT)):
                self.assertIsNone(definition.search(text), "test body belongs in tests/native")
                if path == SRC / "app" / "cli.cpp":
                    continue  # Legacy --self-test/reference command dispatch.
                for name in INCLUDE.findall(text):
                    target = include_target(path, name)
                    self.assertFalse(
                        target is not None and NATIVE_TESTS in target.parents,
                        f"production module includes test header {name}",
                    )

    def test_obsolete_root_fragments_are_removed(self):
        self.assertEqual(list(SRC.glob("*.inc")), [])
        for name in ("block_jacobi_preconditioner.cuh", "pcg_spmv_layout.cuh",
                     "cudss_direct_solver.cuh"):
            with self.subTest(file=name):
                self.assertFalse((SRC / name).exists(), "header belongs in src/solvers")
                self.assertTrue((SRC / "solvers" / name).is_file())

    def test_module_headers_are_guarded(self):
        headers = [path for path in source_files(SRC) + source_files(NATIVE_TESTS)
                   if path.suffix in {".h", ".cuh"}]
        self.assertTrue(headers, "expected solver module headers")
        for path in headers:
            text = source_text(path).lstrip()
            with self.subTest(file=path.relative_to(ROOT)):
                guarded = re.match(r"#\s*pragma\s+once\b", text)
                if not guarded:
                    guard = re.match(r"#\s*ifndef\s+(\w+)\s*\n\s*#\s*define\s+\1\b", text)
                    self.assertIsNotNone(guard, "header needs an include guard")
                    self.assertRegex(text, r"#\s*endif\b[^\n]*\s*\Z")

    def test_local_include_targets_resolve(self):
        for path in source_files(SRC) + source_files(NATIVE_TESTS):
            for name in INCLUDE.findall(source_text(path)):
                if path == SRC / "native_airgap_matrix.cpp" and name == GENERATED_AGE:
                    continue  # CMake writes the FEMM-derived matrix into the build directory.
                with self.subTest(file=path.relative_to(ROOT), include=name):
                    self.assertIsNotNone(include_target(path, name), "missing local include")


if __name__ == "__main__":
    unittest.main()
