# TUAS South Endfeed — Application State & Preset-ROI Extraction Report

_Last updated: 2026-06 • branch `v1.2` • `program_version = "1.2"`_

This report has two parts:
1. **Current state of the application** — what is built, what is production, what is in test.
2. **Preset-ROI chassis extraction** — the complete process and the reasoning behind every
   step, including the two earlier designs that failed and why.

---

# Part 1 — Current State of the Application

## 1.1 What the application does
A downward-looking SICK Visionary-T-Mini ToF camera per lane guides an automated
container-terminal crane spreader placing/removing containers onto truck chassis. The app
fuses **2D object detection (YOLO via TensorRT)** with **3D point-cloud geometry (PCL /
Open3D)** and reports four outcomes to a PLC over a raw TCP byte protocol:
1. **Landout / Land-OK** (mount cycle)
2. **CLPS** — Chassis Lift Prevention (offload cycle)
3. **CLPS-OK** — clear container/chassis separation (offload cycle)
4. **3D deviation** between container and chassis (mm)

Chassis types: **XT** (external, cone target), **IGH** (internal, cone + guide), **CST**
(internal, guide only). Target appearance varies by camera position (Front/Center/Rear),
job position (Big/Small/Center) and chassis length (20/40/45 ft).

## 1.2 Pipelines
`ProcessLogic()` runs two pipelines per frame and combines them:
- **VA (2D)** — TensorRT YOLO inference → `target_selections_va` → `deviation_output_va` →
  `landOut_detected_va` / `clps_detection_va`.
- **PCA (3D)** — target point-cloud extraction → `position_detection_pca` →
  `deviation_output_pca`.
- Combined: `bLandOut/bClps/bClpsOk = VA || PCA`, with N-count debouncing.

## 1.3 Inference backend (decided)
- **TensorRT is the production runtime** (`TrtRunner`, `.engine`). The active model is
  `model/20251201_TUAS_SEndfeed_3DSP.engine`.
- **ONNX is the shippable model artifact**; the `.engine` is a **per-machine cache** built
  on the target RTX 50-series box (commissioning step / build-if-missing). Engines are
  locked to GPU arch (sm_120) + TRT version. Compiling the C++ app needs no GPU.
- ONNX-Runtime (`VA`) and `YOLO11Detector` paths exist but are commented out.

## 1.4 What is NEW / in active development (this effort)
**ROI-based chassis backup** — a VA-independent way to locate the chassis 3D target, for
when the 2D model is unavailable (GPU failure) or performing poorly. Implemented in
**`ChassisRoi.h`** (header-only), integrated in `ProcessLogic`, gated by **`FORCE_ROI_MODE`**
in `process.ini`. Detailed in Part 2.

Status: **validated for XT FRONT on 4 of 5 job configs** (trolley/gantry agreement with the
2D path of ~25–62 mm). A human ground-truth labeling effort is being set up
(`ASSIGNMENT_ROI_groundtruth.md`, `GROUNDTRUTH_GUIDELINE.md`) to measure true accuracy and
drive further tuning.

## 1.5 Threads, protocol, config, modes (unchanged scaffolding)
- **Threads:** `socketClient` (PLC TCP), `thread_tmini_control`, `thread_tmini_data_stream`
  (triple-buffered `DataStack`), `processingThread`, `data_save_thread`.
- **PLC:** `parseCommand()` decodes a 50-byte input frame; `makeCommand()` encodes the
  50-byte output. `NewJobPulse` rising edge resets job state and selects the preset.
- **Config:** external INI (`process.ini`, `seoho.ini`, `IP_LIST.ini`) via `miniINI.h`.
  INI + model files are runtime assets, not in the source tree.
- **Modes:** normal (PLC), `MODE_DEBUG`, `DEBUG_WITH_FILES`, `DEBUG_BATCH_JOB`
  (`OfflineDebugBatchProcessingThread` replays saved jobs → `*_res.txt`).

## 1.6 Build
- Real builds use the Visual Studio `.sln`/`.vcxproj` under `build/` (the `.vcxproj` is
  **hand-maintained**, not regenerated from CMake — add new `.cpp` there manually;
  header-only files like `ChassisRoi.h` need no entry).
- Output: `build/Release/TUAS_Endfeed_3DSP_v1.2.exe`.
- `CMakeLists.txt` cleaned (removed the non-existent `CPS.cpp/CPS.h`; added `ChassisRoi.h`).

## 1.7 Known open items
- `makeCommand()` byte-17 CLPS-OK bit bug (deferred — touches live PLC output).
- ROI: **CHS20-Big** config needs its own preset key; **height (H)** accuracy is looser
  than trolley/gantry; **REAR** and **IGH/CST** not yet validated with the ROI path.
- Pre-existing CLPS/CLPS-OK/landout logic divergences tracked in `CLAUDE.md`.

---

# Part 2 — Preset-ROI Chassis Extraction

## 2.1 The problem it solves
The 3D target-finding originally depended entirely on the 2D detector: VA finds a bounding
box, the box is mapped to the point cloud, and `pc_*_detection_naive` isolates the target
within that box. **If the 2D model is down (dead GPU) or blind (bad lighting / poor
performance), there is no 3D result at all.** We need a model-free path that finds the same
chassis target from the depth cloud alone.

## 2.2 The two enabling facts (domain + sensor)
1. **The chassis is "in-place" when a job is enabled.** Lane borders constrain the chassis
   so its target sits within **±200 mm of an ideal trolley position** that is known per job
   configuration. So a per-job **world-space Region of Interest (ROI)** can stand in for the
   VA box.
2. **The point cloud is organized 512×424**, the same grid as the sensor image
   (`index = y*512 + x`). World coordinates are **T (trolley, x), H (height, y, up = +),
   G (gantry, z)**, in millimetres, after a tilt transform. Left/right runs along **G**
   (sensors are mounted on the lane side).

## 2.3 The chassis is geometrically special: the cone protrudes toward the sensor
The single most important physical fact for extraction: the XT twist-lock **cone sticks up
toward the side-mounted sensor**, so **it is the nearest-Z structure** in its neighborhood.
The chassis deck behind/around it is farther in Z; the wheel/hub sits well *below* it in
height (≈ 480 mm lower). This is what lets a model-free method pinpoint the cone.

## 2.4 The process (current, validated design)

Gated by `FORCE_ROI_MODE=1`. Per frame, for the chassis target:

**Step 0 — Pick the per-job preset ROI.** On the New-Job pulse, the app selects a preset by
(sensor, chassis length, job position, spreader size). The preset supplies a **trolley
range** (`presetRangeX`, INI `PRESET_FRONT_3D` / `PRESET_REAR_3D`, format `MAX;MIN`). Height
and gantry bounds are global (`HEIGHT_RANGE`, `TROLLEY_RANGE`). These three give the world
ROI box: **T ∈ preset (±~210 mm of ideal), H ∈ cone-height band, G ∈ gantry bounds.**

**Step 1 — Crop (`chassis_roi_crop`).** Keep cloud points inside the ROI box; drop invalid
`(0,0,0)` pixels (these are sensor no-returns that otherwise pollute every statistic). Saves
`_<tag>_roiCrop.pcd` for inspection.

**Step 2 — Detect the cone (`chassis_roi_detect_cone`).**
- Compute a **robust nearest-Z** = dense minimum of the crop's Z (a sliding window ignoring
  isolated flyer points).
- Take a **thin near-Z slab**: points with `z ∈ [zNear − zPad, zNear + zSlab]`
  (`zPad = 30`, `zSlab = 100` mm). This slab is the protruding cone, separated from the
  deck behind it.
- The cone position = **centroid of the slab**. Saves `_<tag>_coneSlab.pcd` and
  `_<tag>_conePos.pcd`.

**Step 3 — Track (`ChassisRoiTracker`).** The chassis does not move until load transfer, so
this is **acquire-then-hold**, operating on the detected position (not a re-localization):
- `ACQUIRING` → `LOCKED` after `lockK` (=3) consecutive detections agreeing within `lockTol`
  (=30 mm).
- `LOCKED` holds the baseline; raises **`CHASSIS_MOVED`** (driver creep) after `moveConfirm`
  (=3) consecutive frames drifting beyond `moveTol` (=50 mm), then re-acquires.
- **Frozen** while the spreader is landed (the descending container occupies the cone band).
- Unlocks after `maxMiss` (=15) missed detections.

**Step 4 — Feed downstream.** The detected cone position becomes `_cone_pos` (or
`_guide_pos` for CST), exactly where the VA-derived position used to go, so deviation /
landout / logging are unchanged. The 2D-VA cone path for the chassis is bypassed in ROI
mode (`_pcCone` left empty so `position_detection_pca` skips it); the **hole/container stays
VA-based** (its model-free localization is a later stage).

## 2.5 Why each choice — and the two designs that failed first

The current design is the third attempt. The failures are recorded because they encode the
real geometry.

**v1 — height histogram (FAILED, found the wheel).** Logic: "lowest dense height layer =
deck; cone = 50–250 mm above it." It locked the **wheel/hub** every time, because the cone
is *higher and nearer-Z*, not "just above the lowest layer"; widening the height floor to
−800 also let the wheel and ground into the crop.
→ **Lesson:** the cone is found by **depth (nearest-Z)**, not by a height plane.

**v2 — delegate to `pc_cone_detection_naive` on the preset crop (FAILED on wide crops).**
That detector uses a **deep 300 mm Z-slab + trolley-extreme corner**, which is correct for
VA's *tight, cone-centered box*. But the chassis top is a **continuous structure at cone
height spanning the whole trolley range**; on a wide preset crop the deep slab is
deck-dominated and the corner grabbed the wrong trolley edge (e.g. returned T≈943 vs true
666). Gantry was already accurate (~21 mm); only trolley was wrong.
→ **Lessons:** (a) a **thin** near-Z slab is essential — a deep slab lets the deck win; data
showed nearZ+50/+100 → cone (T≈690), nearZ+150/+300 → deck (T≈800–850). (b) The preset
**trolley range must be tight and cone-centered** so the target cone — not a neighbor cone or
the partner across the chassis — is the in-range nearest-Z structure.

**v3 — nearest-Z thin slab, loose ranges (PARTIAL).** Fixed the configs whose range was
already reasonable but regressed the wide-range ones (nearest-Z picked a different protrusion
elsewhere in the wide crop). Confirmed the range-tightness requirement.

**v4/v5 — nearest-Z thin slab + tight (±210 mm) ranges (CURRENT, validated).** With each
config's trolley range centered on its true cone within the ±200 mm lane bound, the thin
near-Z slab lands on the target cone.

**Why nearest-Z + thin slab + tight range together:** the thin slab removes the deck (depth
separation); the cone-height H band removes the wheel/ground (height separation); the tight
trolley range removes neighbor/partner cones (lateral separation). Each axis rejects a
different confuser; all three are needed.

## 2.6 Validation (XT FRONT, vs the 2D path's cone, 2,916 frames)

| Config | trolley \|dT\| median | within 100 mm |
|---|---|---|
| Mount Small CHS40 | 25 mm | 98% |
| Mount Center CHS45 | 35 mm | 89% |
| Mount Center CHS40 | 62 mm | 93% |
| Mount Big CHS40 | 61 mm | 83% |
| Mount Big CHS20 | 302 mm | 26% (needs own preset key) |

Overall: trolley median 57 mm, **gantry median 33 mm**, 77% within 100 mm trolley. Height
(`dH`) median 143 mm — looser because the slab-centroid height differs from the corner
target height; it does not affect lateral alignment and is deferred.

Note: this compares ROI against the **2D path**, which is itself imperfect. The human
ground-truth effort will give absolute accuracy.

## 2.7 Real-time budget
Target ≤ 300 ms/frame (current loop ≈ 200 ms). The ROI extraction is a single ROI
passthrough + a 1-D nearest-Z computation + a centroid — order ~1–2 ms — well within budget
and far cheaper than the voxel/cluster path it replaces.

## 2.8 Configuration (`process.ini`)
- `[SETTINGS] FORCE_ROI_MODE=1` enables the path. `HEIGHT_RANGE_MAX/MIN` = cone-height band
  (currently 200 / −600, excludes the wheel). `TROLLEY_RANGE_MAX/MIN` = gantry bounds
  (2150 / 1450). Optional tracker keys: `ROI_LOCK_K`, `ROI_LOCK_TOL`, `ROI_MOVE_TOL`.
- `[PRESET_FRONT_3D]` / `[PRESET_REAR_3D]` = tight per-config trolley ranges (`MAX;MIN`),
  ideal ±210 mm, derived from observed cone positions.

## 2.9 Debug artifacts (per frame, batch mode)
`_<tag>_roiCrop.pcd` (searched crop), `_<tag>_coneSlab.pcd` (near-Z slab), `_<tag>_conePos.pcd`
(ROI cone point; tag = `ROI_TRK` locked / `ROI_ACQ` acquiring), `_AB_va_body.pcd` +
`_AB_va_conePos.pcd` (2D-VA box mapped to 3D + its cone point, for A/B), `_base_converted.pcd`
(full scene). `res.txt` adds `ROI_State;ROI_T/H/G;ROI_CropPts;AB_Valid;AB_dT/dH/dG`. Eval:
`tools/roi_eval.ps1`; PCD inspector (numpy-only): `tools/pcd_inspect.py`.

## 2.10 Next steps
1. Dedicated preset key for **CHS20-Big** (its cone falls outside the CHS40-centered range).
2. Improve **height** accuracy (corner-anchored, not slab-centroid).
3. Validate **REAR**, then **IGH** and **CST** chassis.
4. Calibrate against **human ground truth** when it arrives; then extend the model-free
   approach to the **container corner-casting (hole)**.
