# TECHNICAL SPECIFICATION  
## Reconstruction of FDM-Printed Parts for Structural Analysis and Geometry Recovery  
### Project: slice2solid

---

## 1. Purpose of the Project

The purpose of this project is to develop a universal software tool for reconstructing
a digital representation of a part manufactured by **FDM (Fused Deposition Modeling)**,
based on slicer-generated data (Stratasys Insight),
with subsequent use of the results for:

- structural (CAE / FEA) analysis,
- recovery of explicit internal geometry (infill),
- usage of the reconstructed part in CAD assemblies and downstream engineering tasks.

The project targets **engineering-level accuracy** and is intended for use with
finite element software such as **ANSYS**.

---

## 2. General Concept

The project implements a **dual-representation workflow** based on the same slicing data:

1. **CAE-Oriented Representation (Primary Mode)**  
   Used for structural analysis.  
   The geometry is smooth and closed, while the internal FDM structure is accounted for
   through material anisotropy and toolpath-based orientation fields.

2. **Geometric Reconstruction Mode (Optional, Post-CAE)**  
   Used to explicitly reconstruct the internal infill geometry
   after CAE results have been validated.
   Intended for CAD assemblies, mass and inertia evaluation,
   visualization, and archiving.

Both modes rely on the **same toolpath and slicing data**,
but apply them for different engineering purposes.

---

## 3. Manufacturing Workflow Description

### 3.1 CAD Stage
1. A monolithic solid part is designed in a CAD system.
2. The geometry is exported as an STL file in the CAD coordinate system.

### 3.2 Slicing Stage (Stratasys Insight)
1. The STL file is imported into Stratasys Insight.
2. The part is positioned and oriented on the build plate.
3. The oriented part is saved as a new STL file (placed STL).
4. Printing parameters are defined (including infill percentage and pattern).
5. Slicing is performed.
6. Toolpath simulation data is exported.

---

## 4. Project Input Data

### 4.1 Geometric Data
- Original STL file (from CAD)
- STL file after placement in the slicer (placed STL)

The placed STL is considered the **authoritative geometric reference**
for all subsequent stages.

---

### 4.2 Toolpath Data (Primary Source)

Text-based toolpath simulation export containing, for each segment:
- X, Y, Z coordinates,
- time stamps,
- bead cross-sectional area,
- segment type,
- bead / printing mode.

This data represents the **manufacturing intent** and is the primary source for:
- internal structure definition,
- toolpath directions,
- layer organization.

---

### 4.3 Slicer Metadata (Secondary Sources)

Additional slicer-generated files may contain:
- shrink factors,
- coordinate transformation matrices,
- slicing parameters,
- machine and build mode information.

These files are used to refine interpretation of the toolpath data.

---

### 4.4 Stratasys Insight Job Artifacts (ssys_* Folders)

For Stratasys Insight, slicing results are commonly stored in a job folder (e.g., `ssys_part`, `ssys_part-table`, `ssys_g-part`),
which may include:

- `*.sjb` — text job descriptor (paths to STL, job name, toolpath filename).
- `*.cmb` — binary toolpath container used by the machine.
- `*.sgm.gz` — compressed text describing slice/group structure (e.g., Base/Part/Support groups).
- `*.sbs` — build style script (material, tips, slice height and other settings).
- `sliceParams.*`, `toolpathParams.*`, `supportParams.*` (`.cur` / `.new`) — text parameter snapshots (Tcl `setVersionVar ...`).
- `log.txt`, `graphics.log`, `*.bmp` — diagnostics/preview artifacts.

These files are treated as **secondary metadata**. The primary input remains the text-based toolpath simulation export.

---

### 4.5 Verified Classification Rules (Insight Simulation Export)

Based on multiple experimental exports, the `Toolpath Simulation Data` table in the simulation export can be reliably interpreted as:

- `Type = 1` — model (part) toolpath.
- `Type = 0` — support/base toolpath (includes build base/raft and supports in volume when present).

For separating build base/raft from supports:

1. Compute `z_model_min = min(Z) over rows where Type = 1`.
2. **Build base / raft**: rows where `Type = 0` and `Z < z_model_min`.
3. **Supports (non-base)**: rows where `Type = 0` and `Z >= z_model_min`.

`BeadMode` values are **configuration-dependent** for model toolpaths (`Type = 1`) and should not be hardcoded to specific meanings.
However, in tested datasets, the lowest base layers consistently used stable `BeadMode` codes (e.g., `107`/`109`) while supports in
volume introduced additional `BeadMode` values (e.g., `200`/`202`/`205`/...).

---

## 5. Coordinate Systems and Transformations

The following coordinate systems are involved:
1. CAD coordinate system
2. Slicer / build plate coordinate system
3. CAE coordinate system

All toolpath data is transformed into the coordinate system of the placed STL,
using slicer-reported transformation matrices and shrink factors.

---

### 5.1 Transformation Convention (Insight Simulation Export)

The simulation export contains:

- per-axis shrink factors (model/support),
- a `STL to CMB transformation matrix`,
- a toolpath table of points/segments in **CMB coordinates**.

The placed STL coordinate system is treated as authoritative. Therefore, when mapping toolpath points to the placed STL:

- toolpath `XYZ` is read in CMB coordinates,
- the inverse transform (`CMB -> STL`) is applied using the provided `STL to CMB` matrix (and shrink factors as reported by the slicer).

Implementations must validate the mapping by checking that transformed toolpath points lie within (or near) the placed STL bounds.


## 6. CAE-Oriented Representation (Primary Mode)

### 6.1 Geometry
- Smooth, closed solid geometry
- Conforming to the placed STL
- No explicit modeling of infill, supports, or raft

### 6.2 Mesh
- Standard volumetric finite element mesh (tetrahedral or hexahedral)
- Mesh density independent of individual bead geometry

---

## 7. Material Model

The printed material is modeled as **transversely isotropic**:

- Plane of isotropy: printing plane (XY)
- Axis of anisotropy: build direction (Z)

Material parameters include:
- Elastic moduli: E₁ = E₂ ≠ E₃
- Shear moduli: G₁₂, G₁₃, G₂₃
- Poisson ratios: ν₁₂, ν₁₃, ν₂₃

---

## 8. Treatment of Internal Infill Structure

The internal infill structure generated during slicing
(infill percentage, pattern, and toolpath orientation)
is **not reconstructed as explicit geometric voids**
in the CAE-oriented representation.

Instead, the effect of infill is captured through:
- toolpath-based material orientation,
- assignment of local material coordinate systems,
- effective anisotropic material properties.

Thus, a smooth solid geometry is used as the computational domain,
while the internal FDM structure is represented physically
through material behavior rather than explicit geometry.

---

## 9. Toolpath-to-Element Mapping

Each finite element is assigned a **local material coordinate system**:

- Local X-axis: aligned with the dominant toolpath direction,
- Local Z-axis: aligned with the build direction,
- Local Y-axis: orthogonal completion.

Toolpath directions are derived from:
- consecutive toolpath points,
- layer segmentation based on Z-coordinate.

---

## 10. Supports and Raft

- Support material and raft layers are identified by toolpath type
  and/or Z-level filtering.
- These regions are **excluded** from the structural geometry.
- Support data may be retained for diagnostic purposes only.

---

## 11. Explicit Geometry Reconstruction (Optional Mode)

After completion and validation of CAE analysis,
the project provides an optional mode for reconstructing
the **explicit internal geometry** of the printed part.

Geometric reconstruction may be performed using:
- bead sweep along toolpaths,
- voxel-based reconstruction followed by solid generation,
- hybrid approaches combining a smooth outer shell
  with explicit internal infill geometry.

This mode is intended for:
- use in CAD assemblies,
- mass and inertia calculations,
- visualization and documentation.

---

### 11.1 nTop (nTopology) Workflow (Recommended for Solid Conversion)

Direct conversion of voxel/infill geometry into a valid B-Rep solid (STEP) can be computationally expensive and unstable.
Therefore, the recommended workflow for generating a CAD-ready solid with internal voids is:

1. Reconstruct a volumetric representation of the printed material from `Type = 1` toolpaths (model only).
2. Export the volume as a volumetric field file suitable for nTop (recommended: **OpenVDB**).
3. Convert the volumetric field into a solid inside nTop using its internal tools (isosurface/solidification/repair).
4. Export the final CAD solid as **STEP** from nTop.

The project shall therefore provide:
- `*.vdb` (primary) — volumetric field representing material/void.
- `*.stl` (optional) — preview mesh generated from the volume.
- `*.json` — metadata (voxel size, thresholds, source job identifiers, shrink/matrix info, toolpath filtering rules).

---

## 12. Project Outputs

The project generates:
- CAE-ready solid geometry (STL / STEP),
- toolpath-derived material orientation data for ANSYS,
- explicit infill geometry (optional),
- diagnostic and statistical reports.

---

## 13. Implementation Phases

### Phase 1 — Slicing Data Interpretation
- parsing of slicer-generated files,
- coordinate system validation,
- classification of toolpaths.

### Phase 2 — CAE Representation
- generation of smooth solid geometry,
- assignment of anisotropic material orientations,
- export for finite element analysis.

### Phase 3 — Geometry Reconstruction
- explicit infill generation,
- solid body construction,
- export for CAD and assembly usage.

---

## 14. Assumptions and Limitations

- Porosity is accounted for through effective material properties.
- Thermal residual stresses are not modeled.
- Material properties are assumed homogeneous within oriented regions.

---

## 15. Possible Extensions

- Interlayer cohesive zone models,
- progressive damage modeling,
- residual stress simulation,
- multi-material FDM printing.
