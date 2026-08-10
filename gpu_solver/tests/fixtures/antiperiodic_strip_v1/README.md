# Anti-periodic strip reference v1

This fixture matches `periodic_strip_v1` except that the left and right
boundaries are anti-periodic. Stock FEMM therefore enforces equal-magnitude,
opposite-sign nodal potential on every paired boundary node.

The frozen stock FEMM answer, circuit properties, and immutable generic
artifact are checked by the standalone CLI regression. Nodal potential and
circuit flux linkage must match within the 1e-10 relative limit, and every
anti-periodic pair is checked directly. `freeze_reference.lua` documents the
CPU reference procedure; the stock installation remains read-only.
