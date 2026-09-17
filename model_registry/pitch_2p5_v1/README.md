# pitch_2p5_v1

This directory is the canonical registry copy of the formally delivered
`pitch_2p5_v1` checkpoint and frozen inference profile (delivery date
2026-09-14). Both copied files are verified by SHA-256 in
`model_metadata.json`.

The model is a pilot specialized model for 2.5 um arrays in the current
validated imaging domain. It is not the general default and must not be used
for other pitch, microscope, scale, exposure, or sample conditions without
representative manual QC.

The delivery profile contains a fixed `model_space.scale` and explicit
component-filter bounds. Integrations must honor the complete profile; merely
swapping the checkpoint is not a valid upgrade. Until the profile adapter and
real-image regression are accepted, Workbench keeps `unified_v2` as the
default model and exposes this release as a candidate.
