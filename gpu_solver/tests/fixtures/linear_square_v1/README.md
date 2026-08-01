# Frozen linear FEMM reference v1

This is a small planar DC, linear-mu, PM-free FEMM 4.2 reference problem. A
12 A series circuit uniformly excites a 1 m by 1 m, 1 m deep region with
relative permeability 1 and zero-A Dirichlet conditions on the outer square.
Smart meshing is disabled. The frozen solution has 35 nodes and 48 P1
triangles.

`linear_square.ans`, the circuit export, and the matrix dump were generated
with the official FEMM 4.2 22Oct2023 x64 distribution extracted into a scratch
directory. The repository's stock CPU paths and `C:\femm42` were not modified.
The comment `dump` causes fkn to write the internal algebraic system as `.m`
and `.dat`; the `.ans` nodal values are physical A in Wb/m.

Scope exclusions: nonlinear B-H data, permanent magnets, AC/complex values,
axisymmetry, periodic/air-gap constraints, and voltage-driven circuits.

