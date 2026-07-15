# ROI Ground-Truth Validation — Session Handoff (2026-06-22)

Hand-off note so another session (Claude desktop) can continue the chassis ROI
target-position work. Read this together with `CLAUDE.md` (project memory),
`PRESET_ROI_REPORT.md` (feature + extraction overview), and
`GROUNDTRUTH_GUIDELINE.md` / `ASSIGNMENT_ROI_groundtruth.md` (the labeling effort).

---

## 1. What this session did

First batch of **human ground truth** arrived. We evaluated the ROI chassis-cone
target-position logic (`chassis_roi_detect_cone` in `SampleVisionaryTMini/ChassisRoi.h`)
against it, **without raw frames** — only the human-cropped cone bodies + hand-picked
target points were available.

### Ground-truth data location
`K:\TUAS\South_Endfeed\GroundTruth_Set\groundtruth_20260619\`
- `groundtruth.txt` — `작업폴더명 ; 프레임명 ; 상태 ; X ; Y ; Z` (`;`-separated; X,Y,Z =
  hand-picked target = **T,H,G mm**). 66 rows = 11 jobs × 2 sensors × 3 states.
- One folder per job/sensor, each with `Chassis/`, `Separated/`, `Landed/` subfolders
  holding the human-segmented cone-body PCD for that state.
- States: **CONLY** (`Chassis/`, cone only, no container) → **LANDING** (`Separated/`,
  container descending but still separated) → **LANDED** (`Landed/`, container resting).
- Coverage: XT only; configs = FS/CS/RS × {Center CHS40, Big CHS40, Big CHS20, Small CHS40}.
- PCD format gotcha: these crops are `FIELDS x y z _` with a 4-byte pad (`SIZE 4 4 4 1`,
  `COUNT 1 1 1 4`, stride **16** not 12), `DATA binary`. The reader must compute stride
  from `size*count` and slice xyz by byte offset (handled in the tool below).

### Tool written
`SampleVisionaryTMini/tools/gt_eval.py` — numpy-only. Reads the crops, replicates the
C++ detector exactly (`dense_min` = `chassis_roi_detail::denseExtreme` low-end;
`detect_cone` = nearest-Z thin slab `[zNear−30, zNear+100]` → slab centroid), joins to
`groundtruth.txt`, prints per-frame error and medians by state. Run:
`python SampleVisionaryTMini/tools/gt_eval.py` (defaults to the GT path above).

---

## 2. Operating window (confirmed with site owner this session)

**The chassis target algorithm only needs to be valid in CONLY + LANDING** (cone-only
and container-still-separated). Once **LANDED**, container and chassis target points
cannot be separated even by a human — so LANDED is **out of scope** (do not score or
tune against it).

---

## 3. Key findings (all from `gt_eval.py` on this GT set)

### 3a. Detector (centroid) vs GT, by state
```
state     n   |dT|med  |dH|med  |dG|med   T-bias   H-bias
CONLY    22     54       22       37       -53      -22
LANDING  22     49      143(!)    13       -48      +143
LANDED   22     77       38       16       -76      +38     <- out of scope
```

### 3b. The trolley error is a removable per-config bias (centroid≠corner)
The detector outputs the cone **centroid**; the GT is a specific **corner** (the chassis
plate point where the casting's lower-left lands). The gap is a stable per-config
constant. **CONLY-only, after subtracting a per-config median T offset: |dT| 54 → 6 mm
(p95 18 mm).** Per-config T bias (cen_T − GT_T), CONLY:
```
FS_Center −60  RS_Center −53  FS_Big −52  CS_Big +48  CS_Small −63  RS_Small −57  FS_Big_CHS20 −10  CS_Big_CHS20 +40
```
Corner test confirms root cause: GT_T ≈ the slab's **directional T-edge**, not centroid —
**FS / RS / CS-Small → the +T edge; CS-Big → the −T edge** (same side rule the crane uses,
i.e. existing `target_on_right` / job-position geometry).

### 3c. Chassis is stationary across the window (linchpin for hold strategy)
GT chassis motion **CONLY→LANDING: dT med 12 mm (max 59), dH med 5 mm (max 69)**. It does
not move while the container descends; the −159 mm H sag only appears at LANDED (load
transfer), which is out of scope.

### 3d. Hold-the-lock beats re-detecting during LANDING
Fresh nearest-Z during LANDING latches onto the descending casting (H +143). Options vs
LANDING-state GT:
```
strategy                    |dT|  |dH|  |dG|
plain nearest-Z (fresh)      49    143    13   <- chases casting
height-gated nearest-Z       55     42    40
HELD CONLY lock              52     26    38   <- best + simplest
```
Because the chassis is stationary (3c), **holding the CONLY lock through LANDING gives
CONLY-quality accuracy in both windows** (~6 mm T after corner fix, 26 mm H, 38 mm G).

### 3e. The LANDING "error" is the signal we want
The +143 mm nearest-Z jump during LANDING **is the container corner-casting** (now the
nearest protrusion as it descends). So one nearest-Z pass yields: **chassis = held CONLY
lock**, **casting = fresh nearest-Z during LANDING**, and **deviation = casting − held
chassis**. The H jump is also a clean "container has entered" trigger (mirrors VA
idx1→idx2). Gantry (G) is already excellent everywhere (13–37 mm) and needs no work.

---

## 4. Refined plan (agreed, NOT yet implemented)

All changes scoped to `SampleVisionaryTMini/ChassisRoi.h` + its `ProcessLogic`
integration. **Code changes were authorized for the ROI feature** (still do NOT touch the
deferred `makeCommand()` byte-17 CLPS-OK bug — unrelated, live PLC path).

1. **Corner fix on the CONLY lock (the one real lever).** Replace the slab **centroid**
   with the slab's **directional dense T-edge** toward the casting-landing side (reuse the
   existing side rule: FS/RS/CS-Small → +T edge, CS-Big → −T edge). Keep G as centroid
   (already great); take H at that same edge. Expected chassis ~6 mm T / ~20 mm H / ~37 mm
   G. *Alternative if preferred:* a per-config `[ROI_CAL]` offset table keyed like the
   preset ranges — quicker, but a fitted constant needing re-derivation per config.
2. **Hold through LANDING; do NOT re-verify the chassis with fresh nearest-Z during
   LANDING** (it would chase the casting). Already the tracker's intent (freeze on
   spreader-landed + lock-before-arrival); just ensure no fresh chassis re-detection runs
   once the container is present. Height gate (drop points above the cone band) is the
   fallback only if an independent re-check is ever wanted.
3. **Casting detector (next, needs data).** Treat the LANDING nearest-Z protrusion as the
   container target; validate once the crew labels **casting (hole) GT**. Deviation =
   casting − held chassis, across CONLY+LANDING.

---

## 5. Open items / not yet validated
- Corner fix not implemented yet (awaiting go-ahead on Option A directional edge vs B
  offset table — last question to the user was exactly this).
- This GT validates only the **point-extraction** step (crops are clean, human-tight).
  The **range-selection / wrong-cone** failure (CHS20-Big lumped into the CHS40-centered
  `XT_4XF` key; see `CLAUDE.md` run #4) is a separate problem needing **raw frames** —
  not exercised here. CHS20-Big still needs its own preset key.
- REAR sensor present in GT now (RS configs) and behaves like FS (same +T-edge rule); good
  sign but corner fix should be re-checked on RS after implementing.
- IGH / CST chassis: no GT yet.
- Container corner-casting (hole) GT: not labeled yet — blocks #4.
- Height accuracy looser than T/G (slab-centroid Y ≠ corner Y); corner fix should help;
  refine later.

## 6. Files touched/created this session
- `SampleVisionaryTMini/tools/gt_eval.py` (new) — GT evaluation harness.
- This file (`ROI_GT_VALIDATION_HANDOFF.md`).
- No source/INI changes made this session (analysis only; corner fix pending approval).
