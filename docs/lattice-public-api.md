# Shared lattice contract v1 (candidate)

Use `needle_select.lattice_api` instead of copying lattice.py. It exports the 18 original
functions plus LatticeModel/LatticeSnap as direct aliases: signatures, defaults, output ordering,
error rules and NumPy/SciPy calculation remain unchanged. The calculation source is still
needle_select.lattice; model weights and ROI radius policies are not part of this contract.

`API_VERSION == 1` is the compatibility check. `lattice_source()` reports the actual loaded
calculation file and its SHA-256. Package version 0.2.0 alone does not prove that this candidate
API is present. Deploy the paired candidate checkout; do not install into a running stable environment.

Needle is the unique current maintenance repository. Decoding is a consumer; Live-cell's
historical vendor stays frozen until separately accepted with reproducibility checks.
