# slice2solid

**slice2solid** is an engineering-oriented tool for reconstructing
FDM-printed parts based on slicer data (Stratasys Insight),
with the primary goal of enabling **structural (CAE) analysis**
and optional **explicit geometry reconstruction**.

The project bridges the gap between:
- CAD geometry,
- slicer manufacturing intent,
- and finite element analysis (ANSYS).

---

## Project Motivation

Standard CAD-to-CAE workflows treat FDM-printed parts as isotropic solids,
ignoring:
- printing orientation,
- layer structure,
- toolpath directions,
- infill patterns.

This project provides a methodology and tooling to:
- account for **process-induced anisotropy**,
- map slicer toolpaths to **material orientation fields**,
- perform structurally meaningful CAE simulations,
- optionally reconstruct the **explicit internal infill geometry**
  for CAD assemblies and downstream use.

---

## Key Concepts

The project uses a **dual-representation approach**:

### 1. CAE-Oriented Representation (Primary)
- Smooth solid geometry (based on placed STL)
- No explicit infill geometry
- Internal FDM structure represented via:
  - transversely isotropic material model
  - toolpath-based material orientation
- Intended for structural analysis (ANSYS)

### 2. Geometry Reconstruction Mode (Optional)
- Explicit reconstruction of infill and perimeters
- Based on the same slicer toolpath data
- Intended for:
  - CAD assemblies
  - mass and inertia evaluation
  - visualization and archiving

---

## Typical Workflow

1. Design monolithic part in CAD
2. Export STL from CAD
3. Import STL into Stratasys Insight
4. Orient part on build plate and save *placed STL*
5. Perform slicing
6. Export toolpath simulation data (`*-simulation-data.txt`)
7. Use `slice2solid` to:
   - generate CAE-ready representation
   - map toolpath-based material orientations
8. Perform structural analysis in ANSYS
9. (Optional) Reconstruct explicit infill geometry

---

## Inputs

Minimum required:
- Insight toolpath simulation export (text), e.g. `*-simulation-data.txt` (contains the `Toolpath Simulation Data` header block, `STL to CMB` matrix, and the toolpath table)
- placed STL exported from Insight (required for geometry export/preview)

Optional (recommended for better defaults/metadata):
- Insight job folder `ssys_*` containing `sliceParams.*`, `toolpathParams.*`, `supportParams.*` and related artifacts

Notes:
- The simulation export header is used for `Slice height`, `Segment filter length`, shrink factors, and matrix validation.
- Locale robustness: some exports may use a comma decimal separator.

---

## Repository Structure

```text
slice2solid/
├─ TECHNICAL_SPECIFICATION.md        # Technical specification (EN)
├─ TECHNICAL_SPECIFICATION_RU.md     # Technical specification (RU)
├─ README.md                         # Project overview and usage
├─ requirements.txt
├─ src/                              # Source code
├─ data/                             # Input/output examples (ignored in git if large)
├─ tests/                            # Validation and test cases
└─ .gitignore
