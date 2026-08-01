# Nonlinear PM/coil FEMM reference

Frozen with the unmodified portable FEMM 4.2 22Oct2023 x64 distribution.
No files under `C:\femm42` were read or changed.

The planar DC model uses a 20 mm depth, an outer `A=0` square, the repository
35PN230 29-point B-H table, a linear `mu_r=1.05` permanent magnet with
`Hc=900000 A/m`, and a 5 A series circuit with 50 turns. Smart mesh is off.
The FEMM reference circuit flux linkage is
`8.3345179912279173e-4 Wb`.

The solution has 506 nodes and 890 P1 elements. Each solution element contains
a valid block-label index in `[0,3]`; the solver test must reject missing or
out-of-range labels instead of silently treating them as air. The block labels
map to the explicit steel, PM, coil-air, and default-air regions stored in the
`.fem`/`.ans` files.

The `.m` and `.dat` files are the frozen `fkn` dump. The TSV is an independent
`mo_getelement` export and intentionally contains geometry/group data only,
because that API does not return material names.
