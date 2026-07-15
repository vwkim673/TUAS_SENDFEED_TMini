# CLAUDE.md

Guidance for working in this repository.

## What this project is

This repo began as SICK's `sick_visionary_cpp_samples` (demo code for Visionary-T Mini
ToF cameras) but the **`SampleVisionaryTMini`** target has been turned into a production
industrial vision application: the **"TUAS South Endfeed App"** (`program_version = "1.2"`,
branch `v1.2`).

It guides an automated container-terminal crane spreader placing/removing shipping
containers onto truck **chassis**, using a downward-looking ToF camera per lane. It fuses
**2D object detection (YOLO/TensorRT)** with **3D point-cloud geometry (PCL/Open3D, PCA)**
to produce four outputs, reported to a **PLC over a raw TCP byte protocol**:

1. **Landout / Land-OK** detection (mount cycle)
2. **CLPS** – Chassis Lift Prevention System (offload cycle)
3. **CLPS-OK** detection
4. **3D deviation** between container and chassis (mm)

The other targets (`SampleVisionaryS/T/TVGA`, `SampleAutoIP`) and `sick_visionary_cpp_shared`
(the camera SDK: CoLa 2 control, data streaming, point-cloud generation) are unmodified
SICK scaffolding. **Only `SampleVisionaryTMini/` contains the custom application.**

## Chassis types

- **XT**  – external type, cone target. Model classes `cone`(1) + `land`(2).
- **IGH** – internal type with cone and guide around. Classes `igh`(3) + `ighs`(4).
- **CST** – internal type with guide only. Classes `cstf`(5)/`csts`(6)/`cstb`(7)/`cstsl`(8).

Model classes (9): `{ "hole", "cone", "land", "igh", "ighs", "cstf", "csts", "cstb", "cstsl" }`.
`hole`(0) = container corner-casting / twist-lock hole. Target appearance varies by
**camera position** (Front/Center/Rear), **job position** (Big/Small/Center), and
**chassis length** (20/40/45 ft), handled via the `PRESET_POS` fallback table.

## Detection semantics (domain knowledge — from site owner)

### Class meaning (XT, confirmed against logs)
- `cone`(idx 1) = **unloaded** chassis twist-lock cone shape (no container on chassis).
- `land`(idx 2) = **loaded** state; includes the container corner-castings resting on
  the chassis.
- Offload visual sequence: **landed (`land`/idx2) → lifting → separation → cone-only
  (`cone`/idx1)**. The idx2→idx1 transition *is* the separation signal.

### CLPS — Chassis Lift Prevention (offload cycle)
- Detect **actual lifting of the chassis object** (caused by a missed unlock between
  container and chassis).
- Primary: capture a **baseline chassis object position at job start**; compare the
  current frame's detection to baseline; if the chassis has lifted "enough" → CLPS.
- Secondary: chassis object detected **above a hard Y threshold** → CLPS.
- Fires only when the chassis is deemed lifted.
- **False positive is TOLERABLE** (precautionary). Bias toward sensitivity.

### CLPS-OK (offload cycle)
- Detect **clear container–chassis SEPARATION**.
- Compare current container vs chassis objects; if the **Y difference exceeds a
  threshold** → CLPS-OK. (Equivalently, cone-only / idx1 appearing = container gone.)
- Fires **only on separation with CERTAINTY**; otherwise it is a fault.
- **False positive is INTOLERABLE** — asserting without clear separation can mask an
  actual chassis lift (safety). Bias toward precision; require debounce/certainty.

### PLC Landed bit
- Reliable. Aggregates **all four spreader corners**: all corners off → Off; all on → On.

### Landout / LandOK (mount cycle)
- Mount outcome: container landed correctly on target chassis/guide → **LandOK**;
  incorrectly → **Landout**.
- 2D is **primary but limited in scope**; **3D position detection is the backup and the
  ultimate authority** on landok/landout when detection is valid.

### 3D position detection (mainly mount cycle)
- Detect the target position of each hole and chassis, output **deviation**; aligning the
  two inline = LandOK.
- Hole target point = corner-casting **lower-left corner** (sensor view).
- XT → align to the chassis **plate's** corresponding position.
- IGH → align to the **guide's top-left closest corner** point.
- CST → easiest (more options / wiggle room).

### Re-interpretation of the earlier batch finding
- "CLPS-OK fired in only 1/25 offload jobs" was largely because those recordings stayed
  in the loaded (`land`/idx2) state and **never reached the separation→cone-only phase** —
  not necessarily a logic bug. XT CLPS-OK keying on idx1 is semantically aligned.
- Residual risks to verify against NEW data (landed→lifting→separation→cone-only):
  CLPS-OK paths set `true` with **no N-count debounce** (risky given FP intolerance);
  IGH CLPS-OK hard-requires both `hole`+`cone` (hole unreliable, no fallback); CLPS
  baseline capture (`AVG_CONE`) is order-dependent and can stay unset.
- **New sample data incoming** with full job-state sequences for batch re-examination.

### XT_Batch findings (K:\TUAS\South_Endfeed\ActualJob\XT_Batch, examined 2026-06)
- **Class semantics confirmed in timelines:** mount jobs show clean `cone(idx1) →
  land(idx2)+LandedTrigger` sequences; the visual landed trigger flips exactly at the
  idx1→idx2 transition.
- **Class flapping observed** near landing (single-frame idx2↔idx1 flips, e.g.
  `2L x1 > 1 x1 > 2L x129`). If this happens during offload-landed phase, the XT
  CLPS-OK "cone-only, no hole" path (~line 7735) fires on ONE flapped frame — no
  debounce. **Concrete FP risk for the safety-critical CLPS-OK.**
- **Hole-only landout path is ungated** (`landOut_detected_va` ~7624): fires on
  hole.prob>=85 alone, without any landed/sprd gate — observed firing 111 frames in a
  job where the chassis was never detected once (20260323_181420 Mount Big CHS40).
- A geometry-based landout fired plausibly in 20260324_225100 (cone re-exposed while
  landed, `1L x42` tail) — the XT |coneX-holeX|>w/2 path works.
- **Offload recordings still end while landed** (all `2L`, ~70-frame clips) — the
  separation→cone-only phase is STILL not present in data; CLPS/CLPS-OK remain
  unvalidated. Need recordings spanning TWL-unlock → hoist → cone-only.
- **2025-era recordings** (`*_TMINI_LEFT/RIGHT`, `None_None` naming) are from the older
  project/sensor era: the current model is nearly blind on them (XT 0%). Not
  representative — exclude from metrics.

## Code-vs-semantics audit (CLPS/CLPS-OK/Landout/LandOK, examined 2026-06)

Verdict: implementation PARTIALLY reflects the agreed semantics. Faithful parts:
CLPS baseline-vs-current (VA), CLPS-OK as Y-separation concept (XT/IGH; PCA path),
geometric landout with landed gate + debounce, PCA deviation landout, cycle gating
(CLPS only offload / landout only mount).

### Divergences & bugs (work-list for the build/improve phase)
1. **VA CLPS-OK has NO debounce** (`clps_detection_va` ~7720/7735) — fires on a single
   frame despite FP being intolerable; idx2↔idx1 flapping proven in XT_Batch data.
   Top-priority fix (add N-count + cross-checks).
2. **CST CLPS-OK is semantically wrong** (~7669): fires on `!sprd_landed` for N frames —
   that's "spreader lifted", not "separation observed". In a real chassis-lift (chassis
   dangling with container) all corners unland → CLPS-OK asserts = the intolerable FP,
   structurally guaranteed. Needs redefinition.
3. **LandOK is NOT implemented anywhere** — no flag/PLC bit; "landed correctly" is only
   the absence of Landout. If crane expects an affirmative LandOK, it's a missing feature.
4. **3D is not the authority** — combined results are a flat `VA || PCA` OR (~8955);
   3D never arbitrates/overrides 2D. Agreed intent: 3D decides landok/landout when its
   detection is valid.
5. **Dead PCA code**: `CLPS_Detection_PCA` (:8332, has the hard-Y secondary rule +
   point-count-under-hole), `CLPS_Detection_PCA_Only` (:8143), and
   `Pre_Land_Chassis_Position_PCA` (:8482, `AVG_CONE_PCA` baseline) are defined but
   NEVER CALLED. Active PCA offload path (`deviation_output_pca` :8600) produces
   CLPS-OK only (devY < -300mm, debounced — the only path meeting the certainty bar);
   **CLPS alarm therefore has no 3D backup; CLPS secondary hard-Y rule missing in
   active code.**
6. **Landed gate uses the visual trigger** (`debug_landed_trigger` → `_sprd_landed_trigger`)
   not the reliable PLC `SPRD_Landed` bit; visual flapping resets the landout debounce.
   Plan: PLC bit primary, visual as fallback.
7. **Hole-only landout ungated** (~7624): `hole.prob>=85` alone, no landed/sprd gate,
   latch-only (never releases). Proven firing mid-job with zero chassis detection.
8. **Latent PCA bug**: hole+guide branch of `deviation_output_pca` (~8627-8631) computes
   deviations from `cone_pos` (which is -10000 in that branch) instead of `guide_pos` →
   garbage devY ≈ -10000 satisfies "cone above hole" when landed → near-guaranteed false
   PCA landout for CST/guide mount jobs. Copy-paste fix: use guide_pos.
9. **CLPS baseline robustness** (`pre_land_chassis_position_va`): (a) `AVG_CONE.get_average()`
   used without checking `get_set()` → zero baseline if early frames <70% → spurious CLPS;
   (b) offload baseline capture doesn't filter class index → baseline from `land`(idx2)
   later compared against `cone`(idx1) box (different shape → center_y shift) → spurious
   CLPS after successful separation. Compare same-class only.
10. **Legacy 3D offsets**: `x_offset=14 / z_offset=26` hard-coded "based on TUAS-3DSP"
    (previous project) — must be re-derived for Endfeed geometry; move to INI.
11. Minor: CST landout doesn't reset its count on `!sprd_landed` (XT/IGH do); IGH CLPS-OK
    has no release path and no cone-only fallback; thresholds (prob 70/85, 20px, 50mm,
    5mm/px) are hard-coded — lift to INI.

### Backend decision (agreed)
- **TensorRT is the production runtime**; ONNX is the shippable model artifact;
  **`.engine` is a per-machine cache built on the target RTX 50-series box**
  (commissioning step or build-if-missing at startup). Engines are locked to GPU
  arch (sm_120) + TRT version; compiling the C++ app itself needs no GPU.
  ONNX-Runtime (`VA`) / YOLO11 paths remain as commented-out alternatives.

## Architecture (`SampleVisionaryTMini/`)

- **`SampleVisionaryTMini.cpp`** — ~10k-line single TU. Heart is `ProcessLogic()` which
  runs two pipelines and OR-combines them:
  - **VA** (2D image): `chassis_type_selection_va` → `target_selections_va` →
    `deviation_output_va` → `landOut_detected_va` / `clps_detection_va`.
  - **PCA** (3D cloud): `target_pointcloud_extraction[_preset]` → `position_detection_pca`
    → `deviation_output_pca` (+ `CLPS_Detection_PCA`).
  - Combined: `bLandOutDetected/bClpsDetected/bClpsOkDetected = VA || PCA`, with N-count
    debouncing (`LDO_NCOUNT`, `CLPS_NCOUNT`, per-pipeline `_count` accumulators).
- **Inference backends:** `TrtRunner` (TensorRT, `.engine`) is **the active backend**
  (`model/20251201_TUAS_SEndfeed_3DSP.engine`). `VA` (ONNX Runtime + CUDA) and
  `YOLO11Detector` (`YOLO11.hpp`) exist but their call sites are commented out.
- **Threads** (`main`, non-debug): `socketClient` (PLC TCP), `thread_tmini_control`
  (camera setup), `thread_tmini_data_stream` (frame grab → triple-buffered `DataStack`),
  `processingThread` (runs `ProcessLogic`), `data_save_thread` (async PLY/img/log save).
- **PLC protocol:** `parseCommand()` decodes a 50-byte input frame (heartbeat, crane/lane
  ID, mount/offload cycle, job pos, spreader twistlock/landed, chassis type & length);
  `makeCommand()` encodes the 50-byte output (positions, deviations, detection flags).
- **Config:** external INI files via `miniINI.h` (`parseINI/parseProcessINI/parseIPINI`) —
  thresholds, presets, per-lane camera IPs, tilt, sensor height. INI + model files are
  **runtime assets, not in the source tree**.
- **Modes:** normal (PLC), `MODE_DEBUG`, `DEBUG_WITH_FILES`, `DEBUG_BATCH_JOB`
  (`OfflineDebugBatchProcessingThread` replays saved jobs → `*_res.txt`).
- **Deps:** Windows, CUDA, TensorRT, ONNX Runtime, OpenCV, PCL, Open3D, Boost.

## Build

- Real builds use the **Visual Studio `.sln`** generated under `build/`, not CMake.
- `sick_visionary_cpp_shared.7z` must be extracted into `sick_visionary_cpp_shared/`
  before configuring (top-level `CMakeLists.txt` errors out otherwise).

## Batch result logs (offline evaluation)

Logs live at `K:\TUAS\South_Endfeed\3DSP Log\{FRONT,REAR}\batch_result\`. Each job folder
`{ts}_{Bn}_{cycle}_{jobPos}_{chassisType}_CHS{len}_SPRD{size}_{position}_0` contains
per-frame subfolders (debug img + extracted `.pcd` clouds) and one `{job}_res.txt` CSV
(`;`-separated, 51 columns; header in row 0). Frames where the 2D detector found nothing
emit a **short 27-column row** (VA columns omitted). Key columns: `Hole_prob`(5),
`XT_prob`(11), `Guide_prob`(16), `Landed_Trigger`(21), `Landout_Detected`(22),
`CLPS_Detected`(23), `CLPS_OK`(24), pca `landout/clps/clpsOk`(37-39), `CLPS_OUT`(49),
`CLPS_OK_OUT`(50). `-10000` = sentinel "no value".

## ROI-based chassis backup (FORCE_ROI_MODE, implemented 2026-06)

VA-independent chassis 3D localization for when the 2D model is unavailable (GPU
failure) or blind. Implemented in **`ChassisRoi.h`** (header-only, self-contained),
integrated in `ProcessLogic`; gated by `FORCE_ROI_MODE=1` in `process.ini [SETTINGS]`
(optional keys: `ROI_LOCK_K`=3, `ROI_LOCK_TOL`=30mm, `ROI_MOVE_TOL`=50mm).

- **Method (v3, 2026-06-14 — NEAREST-Z protrusion):** `chassis_roi_crop()` crops the
  per-job preset world ROI (`presetRangeX` trolley × cone-height `HEIGHT_RANGE` ×
  `TROLLEY_RANGE` gantry), drops invalid (0,0,0) pixels. `chassis_roi_detect_cone()` then
  finds the cone as the **NEAREST-Z protrusion**: robust dense min-Z, take a THIN slab
  [minZ−`zPad`, minZ+`zSlab`] (zSlab≈100mm), return its centroid. The twist-lock cone
  sticks up toward the side-mounted sensor, so it is the nearest-Z structure; a thin slab
  isolates it from the continuous deck behind it. `_pcCone`/`_pcGuide` are left empty so
  `position_detection_pca` skips its own cone path; `_cone_pos`/`_guide_pos` are set from
  this detector. Hole stays VA-based.
- **WHY v1/v2 failed (both instructive):** v1 used a height histogram ("lowest dense
  Y = deck, cone 50–250mm above") and locked the WHEEL — the wheel sits ~480mm BELOW the
  cone (cone H≈−240, wheel H≈−720), and widened `HEIGHT_MIN=−800` let it in. v2 delegated
  to `pc_cone_detection_naive` (deep 300mm Z-slab + trolley-extreme corner) — but the
  chassis top is a CONTINUOUS structure at cone height spanning the whole trolley span, so
  the deep slab is deck-dominated and the corner grabbed the wrong trolley edge (ROI T
  systematically high, e.g. 943 vs true 666). **Key data (2026-06-14):** thin near-Z
  slabs localize the cone (nearZ+50 → T med 690, nearZ+100 → 687) but nearZ+150/+300 →
  800–850 (deck). Lesson: the cone is the nearest-Z PROTRUSION; a thin slab is essential,
  a deep one fails. Gantry was already accurate in v2 (med 21mm); only trolley was wrong.
- **Tracking** (chassis stationary until load transfer): `ChassisRoiTracker` now watches
  the DETECTED cone position (output of the detector), NOT a separate localization.
  ACQUIRING → LOCK after K stable detections (±`lockTol`); LOCKED holds the baseline and
  flags `CHASSIS_MOVED` on `moveConfirm` consecutive drift frames > `moveTol`; frozen
  while spreader landed; unlock after `maxMiss` missed detections. Since every frame
  re-runs full detection on a fresh crop, no separate "broad re-verify" is needed.
  Resets in `reset_jobVariables()`.
- **Axis semantics (site owner, 2026-06)**: world frame is T(x)=trolley, H(y)=height
  (up=+), G(z)=gantry. "Left/right" runs along G (sensors on the lane side; G −left→+right
  in camera view; FS/CS −G = chassis front, RS +G = chassis front). NOTE: with the
  delegation design, the existing detector's own `target_on_right` (trolley-corner pick)
  drives the target point; the gantry side-rule (`roi_cone_on_right()`, still defined)
  only matters if a future broad-ROI variant must choose between two cones in one crop.
- In FORCE_ROI_MODE, VA still runs; `[ROI_AB]` log lines + extra res.txt columns
  (`ROI_State;ROI_T/H/G;ROI_CropPts;AB_Valid;AB_dT/dH/dG`) compare the ROI-detected cone
  vs VA's own mapped cone (same detector on the VA box). Debug PCDs per frame:
  `_ROI_{ACQ,TRK}_roiCrop.pcd` (crop handed to detector), `_cone_pos.pcd`,
  `_AB_va_body.pcd`. Authoritative eval: `tools/roi_eval.ps1` joins the ROI run's
  `ROI_T/H/G` to the original VA run's `cone_x/y/z` by frame. PCD inspector (LZF):
  `tools/pcd_inspect.py` (numpy-only).
- **Batch filter**: accepts `FS*`/`CS*` prefixed folders OR folders carrying a
  `_FRONT_`/`_REAR_` sensor token (2026-era naming); excludes `*TMINI*` (other-site/
  2025 era) and anything else. Folder-name parsing locates tokens relative to the
  Mount/Offload token, so prefixed names parse; also accepts `SPRD20` (original code
  only matched the `SRPD20` typo).
- Real-time budget: ≤300ms/frame; steady-state ROI cost ~1–2ms, acquisition ~5–15ms.

### Derived preset ROI values (XT, from XT_Batch VA detections, 2026-06)

From 4,783 valid frames / 18 jobs (`XT_prob≥70`, PCA cone positions). Written into
`build/{Release/}INI/process.ini` (INI format is `KEY = MAX;MIN`). Margins ±~200mm —
only 1–3 jobs per config, widen as data accumulates.

| Preset | Observed target T | INI value (MAX;MIN) |
|---|---|---|
| F_XT_4XF | +164..+911 (3 configs share key) | 1100;-50 |
| F_XT_4XR | −818..−628 (other cone at +793 excluded) | -450;-1050 |
| F_XT_20F | +606..+741 | 950;400 |
| F_XT_20R | **NO DATA** (no FS CHS45+Small job) | -500;-1000 unverified |
| R_XT_4XF | −785..−590 | -400;-1000 |
| R_XT_4XR | +734..+905 | 1100;550 |
| R_XT_20F | +809..+872 | 1050;600 |

Height: FS cone points H −423..+37, RS −507..−277 (deck ~120–150 lower) →
`HEIGHT_RANGE = 200;-800` (union). Gantry: G 1603..2004 → `TROLLEY_RANGE = 2150;1450`.

Mid-job cone switch confirmed (20260325_171202 Mount Big CHS20): detection sat at
T≈190 then jumped to ≈835 when the descending container covered the target cone —
tracker lock + landed-freeze is the defense; without freeze this would be a false
CHASSIS_MOVED.

### FORCE_ROI_MODE test run #1 (XT FRONT, 2026-06-12, → Batch_result_ROI)

12/12 site jobs locked, 96% of frames in LOCK state, lock in 1–4 frames. Landed-freeze
fired correctly. A/B vs VA (1,846 frames): |dT| med 86mm / p95 536mm. Per-config tight
presets agree well (Mount Center CHS40: dT med 29mm; Mount Small ×2: 80/85mm); the
star case 20260323_181420 (VA 0% blind) locked successfully. **Three issues found:**
1. **Verify churn**: repeated false CHASSIS_MOVED→re-lock cycles (dG ±60..185) — the
   G-side rule re-runs during verification and flickers between adjacent dense runs.
   Fix: during LOCKED verify, pick candidate NEAREST BASELINE; side rule only at acquire.
2. **Width-pair ambiguity under broad presets**: F_XT_4XF (−50..1100, 3 configs) admits
   both width cones (same G, T ~530mm apart) → Mount Big CHS40 locked the wrong cone
   (T:241 vs VA 771); CHS20 Big lock walked 414→805 as occlusion progressed (lock
   migration the landed-freeze couldn't stop because it happens pre-landed).
   Fix: per-config T sub-ranges and/or T-run splitting + drift-reject vs job-start position.
3. **Ground-vs-deck**: `HEIGHT_RANGE_MIN=−800` admits ground (~−780); "lowest dense Y
   mode" latches onto it in most jobs (deck:−779..−780 signatures) → H errors (A/B |dH|
   med ~250–450 on ground-deck jobs vs 136 on the one correct-deck job). T mostly
   unaffected. Fix: deck candidate must have a valid cone layer above it (iterate Y modes
   from lowest), or per-sensor deck band in INI.

### FORCE_ROI_MODE test run #2 (XT FRONT, 2026-06-13, delegation design)

Eval `tools/roi_eval.ps1` (ROI cone vs original VA cone, 2,916 matched frames):
overall |dT| med 268, |dG| med **21**, |dH| med 188. **Bimodal by config:**

| Config | \|dT\| med | ≤100mm | ROI T vs VA T |
|---|---|---|---|
| Mount Center CHS40 | **10** | **99%** | 834 vs 837 ✓ |
| Mount Center CHS45 | 257 | 18% | −540 vs −636 |
| Mount Big CHS40 | 313 | 25% | 1046 vs 819 |
| Mount Small CHS40 | 263 | 0% | 925 vs 661 |
| Mount Big CHS20 | 563 | 0% | 1002 vs 478 |

**Gantry (G) is excellent everywhere (med 21mm)** — chassis is well localized in G/depth.
The failure is purely TROLLEY cone-selection: ROI T is systematically HIGHER than VA T.

**Root cause (confirmed via crop T-histogram):** the preset T-ranges are too WIDE and
contain 2–3 cone-height structures (the target cone PLUS neighbor cones/frame at other
trolley positions). `pc_cone_detection_naive` takes the max-trolley **corner**, so it
returns the high-T edge structure, not the target. Example (Mount Small CHS40, true cone
606–741): crop T 400–950 with cone-height peaks at ~450 / ~660(target) / ~850 → detector
returned T=943. The corner heuristic only works on a **cone-centered single-cone crop**
(which VA's tight detection box always is). Run #1's wheel problem is fixed (H now correct
where T is correct, e.g. Center CHS40); this is the next layer.

**Fix options (needs decision):** (a) tighter cone-CENTERED preset T-ranges — but the
margin to exclude neighbor cones is small (~110mm) and the `4XF` key currently lumps 3
configs at T 478/819/837 (needs finer preset keys / selection split, which also touches
the live PLC path); (b) select the cone-height cluster NEAREST the preset center before
the corner step (robust to neighbors + parking, doesn't need perfect range tightness) —
preferred, but still needs presets roughly centered on the target cone.

### FORCE_ROI_MODE runs #3–#4 (XT FRONT, 2026-06-14, nearest-Z cone)

Run #3 (nearest-Z, loose ranges): |dT| med 106, 49% ≤100mm — fixed some configs but
regressed the wide-`4XF` ones (nearest-Z picked a different protrusion in the wide crop).
**Run #4 (nearest-Z + TIGHT ±210 ranges): |dT| med 57, |dG| med 33, 77% ≤100mm.**

| Config | \|dT\| med | ≤100mm | note |
|---|---|---|---|
| Mount Small CHS40 (20F) | **25** | **98%** | |
| Mount Center CHS45 (4XR) | **35** | **89%** | |
| Mount Big CHS40 (4XF) | 61 | 83% | |
| Mount Center CHS40 (4XF) | 62 | 93% | |
| Mount Big CHS20 (4XF) | 302 | 26% | **cone T~478 excluded by tight 4XF (618-1038); needs own key + has occlusion migration** |

**Method validated for XT**: nearest-Z thin slab + cone-centered ±210 trolley range gives
trolley/gantry agreement with VA of ~25–62mm on 4/5 configs. Remaining: (1) CHS20-Big
needs a dedicated preset key (its cone is outside the CHS40-centered 4XF range); (2) height
(\|dH\| med 143) is looser than T/G because the nearest-Z slab centroid Y differs from VA's
corner-casting Y — matters less for lateral alignment; refine later. Eval:
`tools/roi_eval.ps1 -RoiRoot ...Batch_result_ROI4`.

### Ground-truth labeling (worker, 2026-06)

`GROUNDTRUTH_GUIDELINE.md` (repo root) is the guide handed to a worker to mark human
ground truth for ROI validation. Key facts: the `Image/*.jpg` is **512×424 = the cloud's
organized grid**, so an annotated pixel maps 1:1 to a 3D point (`idx = y*512+x`) — 2D
annotation yields 3D for free. Two CSV schemas: **Tier 1** (per frame/object: box
x1y1x2y2 + target pixel + is_target + visibility + state) on the clean image (NOT the
`_debug_result.jpg`); **Tier 2** (subset, gold 3D target picked in CloudCompare on
`_base_converted.pcd` → T,H,G mm). Frame key = image name w/o extension (joins to res.txt).
Target-point defs: hole=casting lower-left; XT=chassis plate matching corner; IGH=guide
top-left; CST=inner guide corner. Sampling: chassis (stationary) 3 frames/job; hole
(moving) every ~5th frame. When this data lands, join to ROI `res.txt` for true error
stats and resume tuning (CHS20-Big key, dH accuracy, REAR, IGH/CST).

### Debug-batch run setup (dev box)

Exe: `build/Release/TUAS_Endfeed_3DSP_v1.2.exe` (vcxproj is hand-maintained —
`build/SampleVisionaryTMini.vcxproj`, TargetName `TUAS_Endfeed_3DSP_v1.2`, NOT
regenerated from CMake; new .cpp files must be added there manually; header-only
files need no entry). Config `build/Release/INI/seoho.ini`: `DEBUG_BATCH_JOB=1`,
`DEBUG_SENSOR_POSITION=FRONT|REAR`, `DEBUG_BATCH_ROOT_DIR/SAVE_DIR`. The batch
thread **skips jobs whose save folder already exists** — use a fresh save dir per
experiment (e.g. `Batch_result_ROI`). Batch input must be the RAW recording layout
(`<job>/Image/*.jpg` + `<job>/Depth/*.pcd`, e.g. `GroundTruth_Set\XT_20260622`), NOT
the per-frame-folder batch OUTPUT layout. **`ENABLE_MOUNT_REVERSE=1`** ([DEBUG], 2026-07):
every Mount job is processed TWICE — pass 1 forward (normal Mount), pass 2 frames
reversed and treated as OFFLOAD (exercises CLPS/CLPS-OK on the separation sequence);
results split into `SAVE_DIR/Mount_Result/` and `SAVE_DIR/Offload_Result/` (native
Offload jobs go to Offload_Result, no reverse pass). `getAllFiles` now sorts explicitly
so frame order is deterministic. Employee runbook: **`CLPS_REVERSE_BATCH_GUIDE.md`**.

### ROI target-position rework (2026-07) — see ROI_EXTRACTION_PIPELINE.md

Full handoff in **`ROI_EXTRACTION_PIPELINE.md`** (repo root). Status:
- **Region extraction ported to C++** (`ChassisRoi.h: chassis_roi_extract_region`) — matches
  the Python `tools/roi_cone_extract.py` exactly. cy is a **per-config preset** (`roiPrm.coneY`
  from `PRESET_POS` / process.ini `XT_*` 3rd value), NOT runtime (CHS20 wheel is nearer-Z).
- **Target position (NEW, XT)**: ROI region kept FULL; targeting narrows via **upper-half
  (region mid-Y) + near-Z face(80mm)** → `pc_cone_detection_naive` → local-median corner
  height. Vs GT: median |dT|6/|dH|4/|dG|9mm, 72% within 20mm. `pc_cone_detection_naive`
  itself unchanged (input narrowed). OLD (full-region) kept for the 3-way sheet.
- **Tools**: `roi_target_compare.py` (3-way OLD/NEW/GT html), `roi_target_steps.py` (step PCDs)
  → `Comparison\XT_target\`.
- **C++ application complete (2026-07)**: full NEW chain live in ProcessLogic (SOR →
  upper-half+near-face → detector → sparse-edge rescue → corner-height), debug artifacts
  `_ROI_face.pcd`/`_ROI_target.pcd`. CONE_Y = 3rd field of `XT_*` keys in BOTH
  `build/INI/process.ini` and `build/Release/INI/process.ini` (F −218/−148/−274/−274,
  R −438/−323/−435). **parseCY is throw-safe** — legacy comment-as-3rd-field INIs used to
  abort ALL remaining preset parsing via the single try in parseProcessINI (fixed).
  New Center-CHS45 GT samples map to the existing 4XR keys (ranges already cover them);
  the opposite-cone job 150426_451 is NOT coverable by the preset-key scheme (known class).
  Needs VS rebuild (no vcxproj change: header-only + existing cpp).
- **Next (top priority): Part A noise/handle GATE** — near-Z anchor is vulnerable to front
  noise (−120mm). Use density-onset+150mm guard + Y-Z height signature to GATE (not to set
  target Z: height peak = cone centre, 81mm off GT). Then FS_Big residuals, IGH/CST, VS build.

### CLPS / CLPS-OK rework (2026-07) — reverse-replay validated

Study: `tools/roi_offload_replay.py` (+ `roi_clpsok_ab.py`, cache `_replay_cache.csv`) —
Mount jobs replayed BACKWARDS = offload (landed → hoist → separation → cone-only). Real
offload recordings still all end while landed (checked: 2 jobs, all idx2/Landed=1).
Findings (38 jobs): production instant VA cone-only rule fired **while LANDED in 4 jobs
(98 frames, partner cone seen as idx1)** and never fires in 18% (VA blind); while landed
our 3D H reads the CONTAINER corner +22..+169mm above the plate; after separation the
plate H is stable immediately (rebound≈0); ungated stability fires while landed 33/40 →
PLC gate mandatory; re-landing occurs in real jobs (134547) → dynamic gate + release;
devY<−300 unreachable on some configs (max −235) → hole-based conditions demoted.

Implemented (C++):
- **`ChassisRoi.h: ClpsOffload3D`** — 3D CLPS/CLPS-OK state machine. Arm = landed phase
  seen && TWL locked && PLC landed OFF; **CLPS-OK** = H valid && H ≤ landedH+30 (blocks
  "hoist paused with dangling chassis") && H stable stableN frames (≤50mm); latched, re-land
  releases. **CLPS** = H stays ABOVE landedH+30 for liftN frames (plate never re-exposed =
  chassis rising with container). Replay parity: fires land-off +2 med/+4 max frames, 0
  fires while landed. Defaults stableN=12/liftN=15 (~1.2/1.5s @10Hz; replay used 3 at 2s cadence).
- **`_eff_landed`** = PLC `SPRD_Landed` in live / visual trigger in debug (PLC bit was
  parsed-but-unused before). Drives VA CLPS gate + CST rule (site premise: CST has no
  locks → PLC landed OFF alone may assert CLPS-OK).
- **VA CLPS-OK instant-fire removed** (XT cone-only + hole paths): now landed-OFF-gated +
  CLPS_NCOUNT debounced + released on landed re-ON / idx2. IGH same.
- **byte17 fixed**: bits 0/1/2 now the COMBINED bLandOut/bClps/bClpsOk (was VA/VA/PCA mix).
- Combined outputs: `_Out_clps* = VA(gated) || PCA || 3D`.
- **res.txt columns extended** (appended last, prior indices unchanged): after the ROI
  block comes `;EffLanded;CLPS_3D;CLPSOK_3D;LandedH_3D` — the gating landed bit, the
  INDIVIDUAL 3D verdicts (CLPS_OUT/CLPS_OK_OUT are the combined values), and the learned
  loaded-phase H baseline. Lets a batch run show which path fired per frame.
Site premises honoured: landed 중 3D dev 무의미(§1: 그래서 H는 landedH 기준 상대판정),
CST=PLC bit(§3), CLPS arm=TWL_Locked(§6). Next: Landout hole-only PLC-landed gate(§5),
LandOK 별도 로직(§4 — 3D dev로는 부적합), 실제 분리 녹화 확보 시 재검증.

### IGH target positioning (2026-07, PAUSED for XT release) — see IGH_AppState.md

Offline-Python only (not ported to C++). IGH = XT chassis + a GUIDE; target = guide
**top-left corner**. Guide-face rule (`tools/roi_igh_eval.py: target_ighface`) reuses the XT
detector: wide crop → guide-height band (cy=guide TOP, hangs ~200mm BELOW) → near-Z front →
guide(~100mm tilted) + region(near-Z **+300** = guide+plate) → merged-detect → XT-corner with
guide-top height. **3 single-guide configs work** (FS_Big_CHS40/45, CS_Big_CHS45; dT ≤ 11);
**CS_Big_CHS40 is MERGED** (40chassis+20guide joined ~480mm) → target the whole, **offset TBD**
by site owner, GT not a valid ref there. Full state + open items in **`IGH_AppState.md`**.

## Known issues / deferred action items

- ~~`CMakeLists.txt` CPS.cpp/CPS.h references~~ — **removed 2026-06** (target source list
  now includes `ChassisRoi.h` instead).
- **`makeCommand()` byte 17 bug:** bit 2 (CLPS-OK to PLC) is set from
  `PCA_isCLPS_OK` while bits 0/1 use the VA results, and it ignores the combined
  `bClpsOkDetected`. **Needs fixing later** when code changes are next made.
- Code-quality debt: 333 KB single TU, heavy duplication in per-chassis point-cloud
  slicing functions, debug `printf` in the hot inference loop. Stale in-tree copies:
  `20250623_SampleVisionaryTMini.cpp`, `SampleVisionaryTMini.7z`, `z4vg0n0b.ybp~`.

## Working conventions

- The user wants **analysis-only unless explicitly asked to change code.** Do not edit
  source files (incl. the two deferred fixes above) until told to make changes.
