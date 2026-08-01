# Frozen 35PN230 B-H reference v1

`35PN230.tab` is a byte-for-byte copy of
`FEMM_Matlab_Inventor/materials/35PN230/35PN230.tab`. It contains 29 points in
`H [A/m], B [T]` order, including the origin. Its SHA-256 is
`e38e0c2453451e24e9874a96cda2c6363debd8b1a4ac792d4cc932c642f5572d`.

The repository contains no external supplier URL, revision, measurement
condition, or datasheet provenance for this table. The fixture therefore
records repository-local input parity only and must not be described as an
independently verified manufacturer curve.

`generate_reference.m` reproducibly generates `reference_samples.txt` and
`smoothing_reference.txt` in MATLAB R2025a from the DC equations in FEMM's
`CMaterialProp::GetSlopes` and `GetBHProps`. Samples cover every knot and
segment midpoint plus final-slope extrapolation and negative-B symmetry. The
synthetic smoothing reference fixes the processed B/H points and slopes.

The manifest restricts this fixture to raw DC, unlaminated material data:
frequency 0, `LamType=0`, `LamFill=1`, zero conductivity and hysteresis. FEMM's
apparent-curve transformations for other lamination/frequency settings are not
implemented here.

This fixture validates formula/data parity only; it is not an executed fkn
nonlinear solution. It is not a nonlinear mesh solve,
permanent-magnet, force/torque, or performance reference.
