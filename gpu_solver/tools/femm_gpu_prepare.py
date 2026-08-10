"""Create an immutable GPU FEMM artifact from a supported DC ``.fem`` model.

The preprocessor is deliberately separate from the CUDA executable.  It uses
stock FEMM only for its Triangle mesh generator, in a private temporary copy
of the FEMM runtime.  The copied ``fkn.exe`` is replaced by the supplied
``gpu_femm_mesh_noop_solver.exe`` so no CPU magnetic solve is performed and
the installed FEMM directory is never changed.

Supported input is intentionally conservative: DC magnetics; isotropic raw-DC
materials; real series current circuits; constant PM directions; zero-A
Dirichlet boundaries; and FEMM periodic/anti-periodic node pairs. The
preparer emits the legacy planar artifact only when possible, otherwise it
uses the generic magnetostatic artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


class PrepareError(RuntimeError):
    """A model cannot be represented by the current GPU FEMM subset."""


_SETTING = re.compile(r"^\s*\[([^]]+)]\s*=\s*(.*)$")
_PROPERTY = re.compile(r"^\s*<([^>]+)>\s*=\s*(.*)$")
_UNITS_TO_MM = {
    "millimeters": 1.0,
    "meters": 1000.0,
    "centimeters": 10.0,
    "inches": 25.4,
    "mils": 0.0254,
    "micrometers": 0.001,
}


def _atomic_replace(source: Path, destination: Path) -> None:
    """Tolerate short-lived Windows scanner/indexer sharing conflicts."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (2 ** attempt))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        # Stock FEMM commonly writes the active Windows ANSI code page.
        try:
            return path.read_text(encoding="mbcs").splitlines()
        except (LookupError, UnicodeDecodeError) as error:
            raise PrepareError(f"FEMM source must be UTF-8 or Windows ANSI text: {path}") from error


def _unquote(value: str) -> str:
    value = value.strip()
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


def _setting(lines: Sequence[str], name: str) -> str:
    for line in lines:
        match = _SETTING.match(line)
        if match and match.group(1) == name:
            return _unquote(match.group(2))
    raise PrepareError(f"missing FEMM [{name}] setting")


def _number(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise PrepareError(f"invalid {label}: {value!r}") from error
    if not math.isfinite(number):
        raise PrepareError(f"non-finite {label}")
    return number


def _count(lines: Sequence[str], name: str) -> int:
    value = _number(_setting(lines, name), name)
    if value < 0 or value != int(value):
        raise PrepareError(f"[{name}] must be a non-negative integer")
    return int(value)


def _section_start(lines: Sequence[str], name: str) -> int:
    target = name
    for index, line in enumerate(lines):
        match = _SETTING.match(line)
        if match and match.group(1) == target:
            return index + 1
    raise PrepareError(f"missing FEMM [{name}] section")


def _next_content(lines: Sequence[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _property_blocks(lines: Sequence[str], setting: str, begin: str, end: str) -> list[dict[str, Any]]:
    count = _count(lines, setting)
    cursor = _section_start(lines, setting)
    blocks: list[dict[str, Any]] = []
    for item in range(count):
        cursor = _next_content(lines, cursor)
        if cursor >= len(lines) or lines[cursor].strip() != begin:
            raise PrepareError(f"expected {begin} block {item + 1}")
        cursor += 1
        block: dict[str, Any] = {}
        while True:
            cursor = _next_content(lines, cursor)
            if cursor >= len(lines):
                raise PrepareError(f"unterminated {begin} block {item + 1}")
            line = lines[cursor].strip()
            if line == end:
                cursor += 1
                break
            match = _PROPERTY.match(line)
            if not match:
                raise PrepareError(f"invalid property in {begin} block {item + 1}")
            key, value = match.group(1), _unquote(match.group(2))
            if key in block:
                raise PrepareError(f"duplicate {key} property in {begin} block {item + 1}")
            block[key] = value
            if key == "BHPoints":
                points = _number(value, "BHPoints")
                if points < 0 or points != int(points):
                    raise PrepareError("BHPoints must be a non-negative integer")
                curve: list[list[float]] = []
                for _ in range(int(points)):
                    cursor = _next_content(lines, cursor + 1)
                    if cursor >= len(lines):
                        raise PrepareError("truncated B-H curve")
                    values = _numeric_row(lines[cursor], "B-H point")
                    if len(values) != 2:
                        raise PrepareError("each B-H point needs B and H")
                    curve.append(values)
                block["_BH"] = curve
            cursor += 1
        blocks.append(block)
    return blocks


def _required_number(block: dict[str, Any], key: str, label: str) -> float:
    if key not in block:
        raise PrepareError(f"missing {label} property {key}")
    return _number(str(block[key]), f"{label} {key}")


def _required_text(block: dict[str, Any], key: str, label: str) -> str:
    value = str(block.get(key, ""))
    if not value:
        raise PrepareError(f"missing {label} property {key}")
    return value


def _parse_materials(lines: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, block in enumerate(_property_blocks(lines, "BlockProps", "<BeginBlock>", "<EndBlock>")):
        mu_x = _required_number(block, "Mu_x", "material")
        mu_y = _required_number(block, "Mu_y", "material")
        h_c = _required_number(block, "H_c", "material")
        lam_type = _required_number(block, "LamType", "material")
        lam_fill = _required_number(block, "LamFill", "material")
        if mu_x <= 0 or mu_x != mu_y or h_c < 0 or lam_type != 0 or lam_fill != 1:
            raise PrepareError(
                "only isotropic raw-DC materials (Mu_x=Mu_y, LamType=0, LamFill=1) are supported"
            )
        for key in ("H_cAngle", "J_re", "J_im", "Phi_h", "Phi_hx", "Phi_hy"):
            if _required_number(block, key, "material") != 0:
                raise PrepareError(f"material property {key} is not supported by planar DC GPU FEMM")
        curve = block.get("_BH", [])
        b_values = [pair[0] for pair in curve]
        h_values = [pair[1] for pair in curve]
        if len(curve) == 1 or len(curve) > 4096:
            raise PrepareError("a material B-H curve must have zero or 2..4096 points")
        if curve and (b_values[0] != 0 or h_values[0] != 0 or any(
            b_values[i] <= b_values[i - 1] or h_values[i] < h_values[i - 1]
            for i in range(1, len(curve))
        )):
            raise PrepareError("material B-H points must start at (0,0) and be monotonic")
        result.append({
            "id": index,
            "name": _required_text(block, "BlockName", "material"),
            "mu_x": mu_x,
            "mu_y": mu_y,
            "H_c_A_per_m": h_c,
            "B_T": b_values,
            "H_A_per_m": h_values,
            "lam_type": lam_type,
            "lam_fill": lam_fill,
        })
    if not result:
        raise PrepareError("a GPU FEMM model needs at least one material")
    return result


def _parse_circuits(lines: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, block in enumerate(_property_blocks(lines, "CircuitProps", "<BeginCircuit>", "<EndCircuit>")):
        circuit_type = _required_number(block, "CircuitType", "circuit")
        imaginary = _required_number(block, "TotalAmps_im", "circuit")
        if circuit_type != 1 or imaginary != 0:
            raise PrepareError("only real, current-driven series circuits are supported")
        result.append({
            "index": index,
            "name": _required_text(block, "CircuitName", "circuit"),
            "type": "series",
            "current_A": _required_number(block, "TotalAmps_re", "circuit"),
        })
    return result


def _parse_boundaries(lines: Sequence[str]) -> list[str]:
    """Classify supported FEMM boundary markers without losing PBC type."""
    result: list[str] = []
    for block in _property_blocks(lines, "BdryProps", "<BeginBdry>", "<EndBdry>"):
        boundary_type = _required_number(block, "BdryType", "boundary")
        if boundary_type in (4, 5):
            result.append("periodic" if boundary_type == 4 else "antiperiodic")
            continue
        if (boundary_type != 0
                or _required_number(block, "A_0", "boundary") != 0
                or _required_number(block, "A_1", "boundary") != 0
                or _required_number(block, "A_2", "boundary") != 0):
            raise PrepareError("only zero-A Dirichlet and periodic/anti-periodic boundary markers are supported")
        result.append("dirichlet")
    return result


def _numeric_row(line: str, label: str) -> list[float]:
    try:
        values = [float(value) for value in line.split("#", 1)[0].split()]
    except ValueError as error:
        raise PrepareError(f"invalid {label}") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise PrepareError(f"invalid {label}")
    return values


def _parse_labels(lines: Sequence[str], materials: list[dict[str, Any]], circuits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int | None]:
    count = _count(lines, "NumBlockLabels")
    cursor = _section_start(lines, "NumBlockLabels")
    labels: list[dict[str, Any]] = []
    default: int | None = None
    for index in range(count):
        cursor = _next_content(lines, cursor)
        if cursor >= len(lines):
            raise PrepareError("truncated block-label section")
        if re.search(r'"[^"]+"\s*$', lines[cursor]):
            raise PrepareError("spatial PM magnetization formulas are not supported")
        values = _numeric_row(lines[cursor], "block label")
        if len(values) != 9 or any(value != int(value) for value in (values[2], values[4], values[6], values[8])):
            raise PrepareError("each block label must contain nine FEMM values")
        material = int(values[2])
        circuit = int(values[4])
        group = int(values[6])
        flags = int(values[8])
        if material < 1 or material > len(materials) or circuit < 0 or circuit > len(circuits) or group < 0 or flags < 0 or flags > 3 or flags & 1:
            raise PrepareError("block label has unsupported material, circuit, group, or external-region data")
        circuit_index = circuit - 1 if circuit else -1
        turns = values[7] if circuit else 0.0
        if circuit and turns == 0:
            raise PrepareError("a circuit block label must have nonzero turns")
        labels.append({
            "material_id": material - 1,
            "circuit_index": circuit_index,
            "turns": turns,
            "group_number": group,
            "pm_magnetization_deg": values[5],
        })
        if flags & 2:
            if default is not None:
                raise PrepareError("FEMM model has more than one default block label")
            default = index
        cursor += 1
    return labels, default


def _parse_source(path: Path) -> tuple[dict[str, Any], float, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int | None, list[bool]]:
    lines = _clean_lines(path)
    if any(line.strip() == "[Solution]" for line in lines):
        raise PrepareError("input must be a FEMM .fem source, not a solved .ans file")
    frequency = _number(_setting(lines, "Frequency"), "Frequency")
    problem = _setting(lines, "ProblemType").lower()
    units = _setting(lines, "LengthUnits").lower()
    coordinates = _setting(lines, "Coordinates").lower()
    depth = _number(_setting(lines, "Depth"), "Depth")
    if frequency != 0 or problem not in ("planar", "axisymmetric"):
        raise PrepareError("only planar or axisymmetric DC (0 Hz) FEMM models are supported")
    if units not in _UNITS_TO_MM or depth <= 0 or coordinates != "cartesian":
        raise PrepareError("FEMM requires supported LengthUnits, positive Depth, and cartesian coordinates")
    if _count(lines, "PointProps") != 0:
        raise PrepareError("point properties are not supported by GPU FEMM")
    materials = _parse_materials(lines)
    circuits = _parse_circuits(lines)
    labels, default = _parse_labels(lines, materials, circuits)
    if not labels:
        raise PrepareError("FEMM model has no block labels")
    return ({"depth_mm": depth * _UNITS_TO_MM[units], "problem_type": problem, "frequency_hz": 0},
            _UNITS_TO_MM[units], materials, circuits, labels, default, _parse_boundaries(lines))


def _mesh_rows(path: Path, label: str) -> tuple[list[float], list[list[float]]]:
    lines = [line for line in _clean_lines(path) if line.strip()]
    if not lines:
        raise PrepareError(f"empty Triangle {label} file")
    header = _numeric_row(lines[0], f"{label} header")
    count = int(header[0]) if header and header[0] == int(header[0]) else -1
    if count < 0 or len(lines) < count + 1:
        raise PrepareError(f"invalid Triangle {label} header")
    return header, [_numeric_row(line, label) for line in lines[1:count + 1]]


def _indexed_rows(rows: Sequence[list[float]], label: str, minimum_columns: int) -> list[list[float]]:
    indexed: dict[int, list[float]] = {}
    for row in rows:
        if (len(row) < minimum_columns or row[0] != int(row[0])
                or int(row[0]) in indexed):
            raise PrepareError(f"invalid Triangle {label} row")
        indexed[int(row[0])] = row
    if sorted(indexed) != list(range(len(rows))):
        raise PrepareError(f"Triangle {label} IDs must be contiguous and zero based")
    return [indexed[index] for index in range(len(rows))]


def _parse_node_constraints(path: Path, node_count: int) -> list[dict[str, Any]]:
    """Strictly parse stock FEMM's three- or four-column ``.pbc`` records."""
    lines = [line for line in _clean_lines(path) if line.strip()]
    if not lines:
        raise PrepareError("empty Triangle periodic-boundary file")
    header = _numeric_row(lines[0], "periodic-boundary header")
    if len(header) != 1 or header[0] < 0 or header[0] != int(header[0]):
        raise PrepareError("periodic-boundary header must be one non-negative integer")
    count = int(header[0])
    if len(lines) < count + 1:
        raise PrepareError("truncated periodic-boundary file")
    raw: list[tuple[int, int, str]] = []
    for row_index, line in enumerate(lines[1:count + 1]):
        values = _numeric_row(line, "periodic-boundary row")
        if len(values) == 3:
            first, second, kind = values
        elif len(values) == 4:
            record, first, second, kind = values
            if record < 0 or record != int(record):
                raise PrepareError(f"invalid periodic-boundary record ID at row {row_index + 1}")
        else:
            raise PrepareError("periodic-boundary rows must contain 3 or 4 integers")
        if any(value != int(value) for value in (first, second, kind)):
            raise PrepareError("periodic-boundary node IDs and type must be integers")
        first, second, kind = int(first), int(second), int(kind)
        if first < 0 or second < 0 or first >= node_count or second >= node_count:
            raise PrepareError("periodic-boundary node ID is outside the Triangle mesh")
        if first == second:
            raise PrepareError("periodic-boundary pair cannot reference the same node")
        if kind not in (0, 1):
            raise PrepareError("periodic-boundary type must be 0 (periodic) or 1 (anti-periodic)")
        raw.append((min(first, second), max(first, second), "periodic" if kind == 0 else "antiperiodic"))
    tail = lines[count + 1:]
    if tail:
        age = _numeric_row(tail[0], "air-gap count")
        if len(age) != 1 or age[0] < 0 or age[0] != int(age[0]):
            raise PrepareError("invalid air-gap count after periodic-boundary records")
        if age[0] != 0 or len(tail) != 1:
            raise PrepareError("air-gap element data is not supported by the standalone preparer")

    unique = sorted(set(raw))
    by_pair: dict[tuple[int, int], str] = {}
    parent = list(range(node_count))
    parity = [1] * node_count

    def find(node: int) -> tuple[int, int]:
        if parent[node] == node:
            return node, 1
        root, sign = find(parent[node])
        parity[node] *= sign
        parent[node] = root
        return root, parity[node]

    for first, second, relation in unique:
        previous = by_pair.setdefault((first, second), relation)
        if previous != relation:
            raise PrepareError("the same node pair is both periodic and anti-periodic")
        root_a, sign_a = find(first)
        root_b, sign_b = find(second)
        wanted = 1 if relation == "periodic" else -1
        if root_a == root_b:
            if sign_a * sign_b != wanted:
                raise PrepareError("periodic/anti-periodic node constraints are contradictory")
            continue
        parent[root_b] = root_a
        parity[root_b] = wanted * sign_a * sign_b
    return [{"node_a": first, "node_b": second, "relation": relation}
            for first, second, relation in unique]


def _validate_constraint_boundaries(
        constraints: Sequence[dict[str, Any]],
        periodic_nodes_by_marker: dict[int, set[int]],
        boundaries: Sequence[str]) -> None:
    """Bind every `.pbc` pair to the matching FEMM boundary property."""
    paired_nodes_by_marker = {marker: set() for marker in periodic_nodes_by_marker}
    for constraint in constraints:
        first = constraint["node_a"]
        second = constraint["node_b"]
        relation = constraint["relation"]
        matching = [
            marker for marker, nodes in periodic_nodes_by_marker.items()
            if boundaries[marker] == relation and first in nodes and second in nodes
        ]
        if not matching:
            raise PrepareError(
                "periodic-boundary pair is not on one matching FEMM boundary marker"
            )
        for marker in matching:
            paired_nodes_by_marker[marker].update((first, second))
    for marker, nodes in periodic_nodes_by_marker.items():
        if nodes != paired_nodes_by_marker[marker]:
            raise PrepareError(
                "a used periodic/anti-periodic FEMM boundary has unpaired mesh nodes"
            )


def _build_resolved(source: Path, mesh_stem: Path) -> dict[str, Any]:
    model, length_scale_mm, materials, circuits, labels, default, boundaries = _parse_source(source)
    node_header, node_rows = _mesh_rows(mesh_stem.with_suffix(".node"), "node")
    if len(node_header) < 4 or node_header[1] != 2 or len(node_rows) != int(node_header[0]):
        raise PrepareError("Triangle node file is not two-dimensional")
    nodes_by_id: dict[int, list[float]] = {}
    for row in node_rows:
        if len(row) < 3 or row[0] != int(row[0]) or int(row[0]) in nodes_by_id:
            raise PrepareError("invalid Triangle node row")
        nodes_by_id[int(row[0])] = [row[1] * length_scale_mm, row[2] * length_scale_mm]
    if sorted(nodes_by_id) != list(range(len(nodes_by_id))):
        raise PrepareError("Triangle node IDs must be contiguous and zero based")
    element_header, element_rows = _mesh_rows(mesh_stem.with_suffix(".ele"), "element")
    if len(element_header) < 3 or element_header[1] != 3 or element_header[2] != 1:
        raise PrepareError("Triangle element file must contain labeled P1 triangles")
    element_rows = _indexed_rows(element_rows, "element", 5)
    raw_faces: list[tuple[list[int], int]] = []
    for row in element_rows:
        if len(row) < 5 or any(row[index] != int(row[index]) for index in range(5)):
            raise PrepareError("invalid Triangle element row")
        face = [int(row[1]), int(row[2]), int(row[3])]
        if len(set(face)) != 3:
            raise PrepareError("Triangle mesh contains a degenerate element")
        if any(node not in nodes_by_id for node in face):
            raise PrepareError("Triangle element references an unknown node")
        raw_label = int(row[4])
        label = raw_label - 1 if raw_label else default
        if label is None or label < 0 or label >= len(labels):
            raise PrepareError("Triangle mesh contains an unassigned element")
        points = [nodes_by_id[node] for node in face]
        area2 = ((points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
                 - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1]))
        if area2 == 0:
            raise PrepareError("Triangle mesh contains a degenerate element")
        if area2 < 0:
            face[1], face[2] = face[2], face[1]
        raw_faces.append((face, label))

    constraints = _parse_node_constraints(mesh_stem.with_suffix(".pbc"), len(nodes_by_id))
    if model["problem_type"] == "axisymmetric" and any(node[0] < 0 for node in nodes_by_id.values()):
        raise PrepareError("axisymmetric FEMM mesh contains a negative radial coordinate")
    edge_header, edge_rows = _mesh_rows(mesh_stem.with_suffix(".edge"), "edge")
    if len(edge_header) < 2 or edge_header[1] != 1:
        raise PrepareError("invalid Triangle edge header")
    edge_rows = _indexed_rows(edge_rows, "edge", 4)
    triangle_edge_incidence: dict[tuple[int, int], int] = {}
    for face, _ in raw_faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((first, second)))
            triangle_edge_incidence[edge] = triangle_edge_incidence.get(edge, 0) + 1
    if any(count not in (1, 2) for count in triangle_edge_incidence.values()):
        raise PrepareError("Triangle mesh contains a non-manifold edge")
    boundary_nodes: set[int] = set()
    periodic_nodes_by_marker: dict[int, set[int]] = {}
    mesh_edges: set[tuple[int, int]] = set()
    for row in edge_rows:
        if len(row) < 4 or any(row[index] != int(row[index]) for index in range(4)):
            raise PrepareError("invalid Triangle edge row")
        first, second, encoded = int(row[1]), int(row[2]), int(row[3])
        if first == second:
            raise PrepareError("Triangle mesh contains a degenerate edge")
        if first not in nodes_by_id or second not in nodes_by_id:
            raise PrepareError("Triangle edge references an unknown node")
        edge = tuple(sorted((first, second)))
        if edge in mesh_edges:
            raise PrepareError("Triangle edge file contains a duplicate edge")
        mesh_edges.add(edge)
        incidence = triangle_edge_incidence.get(edge)
        if incidence is None:
            raise PrepareError("Triangle edge does not belong to an element")
        if encoded >= 0:
            continue
        if incidence != 1:
            raise PrepareError("Triangle boundary marker is attached to an interior edge")
        marker = -(encoded + 2)
        if marker < 0 or marker >= len(boundaries):
            raise PrepareError("Triangle edge has an invalid FEMM boundary marker")
        if boundaries[marker] == "dirichlet":
            boundary_nodes.update((first, second))
        else:
            periodic_nodes_by_marker.setdefault(marker, set()).update((first, second))
    if mesh_edges != set(triangle_edge_incidence):
        raise PrepareError("Triangle edge file does not match the element topology")
    _validate_constraint_boundaries(constraints, periodic_nodes_by_marker, boundaries)
    if model["problem_type"] == "axisymmetric":
        radius_scale = max(1.0, max(abs(node[0]) for node in nodes_by_id.values()))
        axis_tolerance_mm = 1e-12 * radius_scale
        boundary_nodes.update(node_id for node_id, node in nodes_by_id.items()
                              if abs(node[0]) <= axis_tolerance_mm)
    if not boundary_nodes:
        raise PrepareError("no zero-A Dirichlet boundary nodes were found")

    used = sorted({label for _, label in raw_faces})
    remap = {label: index for index, label in enumerate(used)}
    regions = []
    for index, label_id in enumerate(used):
        label = labels[label_id]
        regions.append({"id": index, "group_number": label["group_number"],
                        "material_id": label["material_id"], "circuit_index": label["circuit_index"],
                        "turns": label["turns"], "pm_magnetization_deg": label["pm_magnetization_deg"]})
    source_sha = _sha256(source)
    resolved = {
        "source_fem_sha256": source_sha,
        "model": model,
        "nodes_mm": [nodes_by_id[index] for index in range(len(nodes_by_id))],
        "triangles": {"node_indices": [face for face, _ in raw_faces],
                      "region_ids": [remap[label] for _, label in raw_faces]},
        "regions": regions,
        "materials": materials,
        "circuits": circuits,
        "outer_dirichlet": {"node_indices": sorted(boundary_nodes),
                            "A_Wb_per_m": [0.0] * len(boundary_nodes)},
    }
    if model["problem_type"] == "axisymmetric" or constraints:
        resolved["node_constraints"] = constraints
    return resolved


def _write_artifact(path: Path, resolved: dict[str, Any], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise PrepareError(f"output already exists (use --overwrite): {path}")
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    document = {
        "schema_version": ("gpu_femm_magnetostatic_mesh_v1"
                           if "node_constraints" in resolved
                           else "gpu_femm_planar_dc_mesh_v1"),
        "source_fem_sha256": resolved["source_fem_sha256"],
        "canonical_identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "resolved": resolved,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        if path.exists() and not overwrite:
            raise PrepareError(f"output already exists (use --overwrite): {path}")
        _atomic_replace(temporary, path)
        temporary = None
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise PrepareError(f"{label} does not exist: {path}")
    return path


def _stage_and_mesh(source: Path, femm_root: Path, noop_solver: Path, timeout_s: float) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    if os.name != "nt":
        raise PrepareError("stock-FEMM mesh preparation is currently supported on Windows only")
    bin_dir = femm_root / "bin" if (femm_root / "bin").is_dir() else femm_root
    _require_file(bin_dir / "femm.exe", "stock FEMM executable")
    _require_file(bin_dir / "triangle.exe", "stock Triangle executable")
    noop_solver = _require_file(noop_solver, "GPU mesh no-op helper")
    temporary = tempfile.TemporaryDirectory(prefix="femm_gpu_prepare_")
    root = Path(temporary.name)
    runtime = root / "runtime"
    shutil.copytree(bin_dir, runtime)
    shutil.copy2(noop_solver, runtime / "fkn.exe")
    working = root / "model.fem"
    shutil.copy2(source, working)
    lua = runtime / "mesh_only.lua"
    lua_path = str(working).replace("\\", "/").replace('"', '\\"')
    lua.write_text(f'open("{lua_path}")\nmi_analyze(0)\nmi_close()\nquit()\n', encoding="ascii")
    try:
        # Pass the path as one raw argument. subprocess supplies Windows command
        # line quoting; embedding quotes here would escape them and FEMM would
        # wait without loading the script.
        run = subprocess.run([str(runtime / "femm.exe"), "-windowhide", f"-lua-script={lua}"],
                             cwd=runtime, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as error:
        temporary.cleanup()
        raise PrepareError(f"stock FEMM mesh generation timed out after {timeout_s:g} s") from error
    if run.returncode != 0:
        temporary.cleanup()
        detail = (run.stdout + "\n" + run.stderr).strip()
        raise PrepareError(f"stock FEMM mesh generation failed ({run.returncode}): {detail}")
    stem = working.with_suffix("")
    missing = [stem.with_suffix(suffix) for suffix in (".node", ".ele", ".edge", ".pbc") if not stem.with_suffix(suffix).is_file()]
    if missing:
        temporary.cleanup()
        raise PrepareError("stock FEMM did not create: " + ", ".join(map(str, missing)))
    return temporary, stem


def prepare(source: Path, output: Path, femm_root: Path, noop_solver: Path, timeout_s: float, overwrite: bool) -> None:
    source = _require_file(source, "input FEMM model")
    output = output.resolve()
    femm_root = femm_root.resolve()
    noop_solver = noop_solver.resolve()
    if output == source:
        raise PrepareError("output artifact must not overwrite the input FEMM model")
    if output == noop_solver:
        raise PrepareError("output artifact must not overwrite the mesh no-op helper")
    protected_femm_root = (
        femm_root.parent
        if femm_root.name.casefold() == "bin" and (femm_root / "femm.exe").is_file()
        else femm_root
    )
    if output.is_relative_to(protected_femm_root):
        raise PrepareError("output artifact must be outside the stock FEMM installation")
    temporary, stem = _stage_and_mesh(source, femm_root, noop_solver, timeout_s)
    try:
        _write_artifact(output, _build_resolved(source, stem), overwrite)
    finally:
        temporary.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="femm_gpu_prepare", description=__doc__)
    parser.add_argument("input_fem", type=Path, help="immutable DC FEMM source model")
    parser.add_argument(
        "output_artifact", type=Path,
        help="new legacy-planar or generic magnetostatic JSON artifact",
    )
    parser.add_argument("--femm-root", type=Path, required=True, help="stock FEMM root or its bin directory")
    parser.add_argument("--mesh-noop", type=Path, required=True, help="gpu_femm_mesh_noop_solver.exe beside the GPU solver")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
            raise PrepareError("--timeout-s must be positive")
        prepare(args.input_fem, args.output_artifact, args.femm_root, args.mesh_noop, args.timeout_s, args.overwrite)
        print(f"Wrote immutable GPU FEMM artifact: {args.output_artifact.resolve()}")
        return 0
    except (OSError, PrepareError, ValueError) as error:
        print(f"femm_gpu_prepare: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
