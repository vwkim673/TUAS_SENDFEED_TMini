# XT Application State — ROI, Target Positioning, CLPS/CLPS-OK (2026-07)

Comprehensive state of the **XT chassis** work before moving on to IGH. Covers the VA-
independent 3D pipeline (ROI extraction → target position), all validation/tests, and the
offload CLPS/CLPS-OK logic. Companion docs: `ROI_EXTRACTION_PIPELINE.md` (pipeline detail),
`CLPS_REVERSE_BATCH_GUIDE.md` (employee runbook), `CLAUDE.md` (project index).

> **Build note:** C++ is not buildable in the dev tooling here (no PCL/CUDA). Real builds
> use the VS `.sln`. `ChassisRoi.h` is header-only (no vcxproj change); only the existing
> `.cpp` was edited. Header is validated by clang syntax-check with a PCL stub.

---

## 1. Purpose & scope

`FORCE_ROI_MODE` (process.ini `[SETTINGS]`) is the **VA-independent backup**: when the 2D
model (YOLO/TensorRT) is unavailable or blind, the ROI path localizes the XT chassis
**cone/plate target** directly from the ToF point cloud and feeds it to the same
`position_detection_pca` / deviation / PLC chain VA uses. XT is complete; IGH/CST are next.

**World frame (PCL):** `p.x` = Gantry (lateral, side-rule axis) · `p.y` = Height (up +) ·
`p.z` = Trolley (depth; the cone protrudes toward the sensor = **nearest-Z**; chassis body
is behind at larger Z). GT `groundtruth.txt` `T,H,G` = `(x,y,z)`.

**XT target point** = chassis **plate corner**: target-X side (left/right by `target_on_right`)
+ nearest-Z + plate-surface height. (Not the cone tip.)

---

## 2. Pipeline (preset crop → target point)

```
preset crop            Gantry cg±300 · Trolley [ct−250,+450] · Height [−600,50]
 │
 ├ candBand = crop @ cy±140          cone-DETECTION layer (excludes handle/wheel by height)
 ├ cone_columns(candBand)            per-X-bin nearest-Z protrusion → cone candidates
 │                                   (protrude_tol 100, min_col_pts 60, X-gap 70 splits cones)
 ├ side-rule                         target_on_right: True→rightmost, False→leftmost cone
 └ extract_region                    tight box around the selected cone, RE-CUT from full crop:
                                     X = nose±150 · Z = front+200 (peak-walk-back) · Y = cone_plate_yband
     │  (= coneRegion; body may be included — targeting below excludes it, not the region)
     ├ SOR                           voxel-hash isolated-flyer removal (k=12, 2σ)
     ├ upper-half                    Y ≥ region mid-Y=(min+max)/2  (drop lower plate/wheel/handle)
     ├ near-Z face                   z ≤ zNear+80  (thin front slab — back body excluded)
     ├ pc_cone_detection_naive       near-Z · target-X corner (x,z)  [UNCHANGED shared detector]
     ├ sparse-edge rescue            face<250pt & supported edge >40mm beyond corner → move to edge col
     └ corner-height                 Y = local median at (x,z) ±60  (plate surface, not slice floor)
 → target (X,Y,Z) → ChassisRoiTracker → deviation → PLC
```

**cy (cone height)** is a **per-config PRESET**, never runtime-measured (the CHS20 wheel is
nearer-Z than the cone, so a runtime nearest-Z estimate grabs the wheel). Python
`PRESETS[key][2]`; C++ `roiPrm.coneY` from `PRESET_POS.*_CY` / process.ini `XT_* = MAX;MIN;CONE_Y`
3rd field. **The "upper-half center-Y" is the region's own (min+max)/2, NOT cy.**

---

## 3. C++ implementation (files & integration)

**`ChassisRoi.h`** (header-only, all validated to match the Python offline logic exactly):
| function | role |
|---|---|
| `chassis_roi_crop` | preset world-ROI crop, drops invalid (0,0,0)/NaN; saves `_<tag>_roiCrop.pcd` |
| `chassis_roi_extract_region` | Stage1–3 (cone_columns / cone_z_extent / cone_plate_yband); `_coneRegion.pcd` |
| `chassis_roi_sor` | **voxel-hash O(n)** SOR (40mm cells, 27-neighbour kNN); flyer removal |
| `chassis_roi_near_face` | upper-half(mid-Y) + near-Z face(+80) prep for the detector |
| `chassis_roi_sparse_edge_rescue` | conditional X-edge correction on sparse faces (+ `denseEdge`) |
| `chassis_roi_corner_height` | local-median plate-surface Y at the detected corner |
| `ChassisRoiTracker` | ACQUIRING→LOCKED baseline, landed-freeze, drift/`CHASSIS_MOVED` |
| `ClpsOffload3D` | 3D CLPS/CLPS-OK offload state machine (§6) |

**`SampleVisionaryTMini.cpp` ProcessLogic (FORCE_ROI_MODE, XT):**
```
_pcCone = extract_region(crop)                    // full region kept (debug/AB)
position_detection_pca(...)                       // OLD result on full region (for 3-way sheet)
── XT targeting ──
  clean = chassis_roi_sor(_pcCone)
  face  = chassis_roi_near_face(clean, 80)         // upper-half + near-face
  nc    = pc_cone_detection_naive(face)
  chassis_roi_sparse_edge_rescue(face, nc)         // conditional
  nc.y  = chassis_roi_corner_height(face, nc.x, nc.z)
  _cone_pos = nc                                   // NEW = production target
  (debug) _ROI_face.pcd / _ROI_target.pcd
── tracker.update(_cone_pos, landed) · A/B vs VA cone ──
```

**INI / presets:**
- CONE_Y = 3rd field of `XT_*` range keys in **both** `build/INI/process.ini` and
  `build/Release/INI/process.ini` (F −218/−148/−274/−274, R −438/−323/−435).
- **`parseCY` is throw-safe** — a legacy free-text comment in the 3rd field used to throw and
  (via the single `try` in `parseProcessINI`) silently abort ALL remaining preset parsing.
  Fixed 2026-07.
- **New preset configs (2026-07):** `FS_Center_CHS45` (−758,1729,−148) and `RS_Center_CHS45`
  (851,1811,−323) added from new GT samples; they map to existing C++ `4XR` keys (T-ranges
  already cover them). The opposite-cone stop job `150426_451` is NOT coverable by the
  preset-key scheme (its target is the other-side cone) — known limitation, same class as the
  CHS20 `215203` stop outlier.

---

## 4. Validation results (vs human GT, 55 CONLY frames / 12 configs)

**Region extraction (vs GT body, 48 frames):** recall 91% · precision 91% · wrong-cone 1/48.

**Target position — final NEW (SOR + upper-half + near-face + rescue):**
- Median error **|dT| 6 · |dH| 4 · |dG| 9 mm**  (OLD full-region detector was dT9/dH10/dG13).
- All-axis within-20mm **41/55 (75%)**, within-30mm 91%, within-40mm 96%.
- New CHS45 configs: FS_Center_CHS45 (3,1,−6) excellent; RS_Center_CHS45 (17,−22,−9) — mild
  H bias from a 127mm job-to-job cone-height spread (per-lane live cy would tighten it).
- **Remaining >40mm outliers (3):** `215203` (ROI-miss: chassis stopped −400mm off preset —
  not a targeting issue), `163000`/`060852` FS_Big_CHS40 (a real structure sits ~40–80mm in
  front of the corner column — geometry can't separate it from the corner; needs temporal).

**Tools (offline, numpy-only):** `roi_cone_extract.py` (extraction), `roi_target_compare.py`
(3-way OLD/NEW/GT html), `roi_target_steps.py` (per-frame step PCDs), `roi_plane_target.py`
& `roi_user_pipeline.py` (rejected experiments, kept for reference), `roi_fullpipe_eval.py`
(GT/cloud resolution + PRESETS), `roi_offload_replay.py` + `roi_clpsok_ab.py` (CLPS study).
Comparison artifacts: `GroundTruth_Set\Comparison\XT_target\` (`index.html` + `steps\`).

---

## 5. Tests / experiments run (what we tried and the verdict)

Everything below was measured on the full GT set — the accepted design is what survived.

**Targeting approaches:**
| approach | result | verdict |
|---|---|---|
| OLD: detector on full 200mm region | dG13; CS_Small dG+60, dH−79 (latches back body) | baseline |
| **NEW: detector on upper-half + near-Z face** | **dT6/dH4/dG9**, body excluded, no regression | ✅ **adopted** |
| edge-aware full redesign (X-edge first) | breaks X (dT 9→36) | ✗ |
| unconditional Z tilt-fix (±60 / ±90 col) | +13~16 bias / cone-tip −40 bias | ✗ |
| conditional **sparse-edge rescue** | exactly 2/47 frames changed, both improved, 0 regress | ✅ **adopted** |
| statistical outlier trim (MAD/percentile, beyond SOR) | CS_Small dG 60→121 (trims the minority target) | ✗ |
| SOR isolated-flyer (voxel-hash) | within-30 41→43, within-40 43→45; no regress | ✅ **adopted** |
| Y-Z height local-max as target-Z | coneZ 81mm off GT (= cone centre, not corner) | ✗ |
| Y-Z height rise-start as front-Z | 13mm off, but near-Z already ≤ that | ✗ (no gain) |
| 3-plane (X-Z/Y-Z/X-Y) reconstruction | within-20 19/46; error cascades across planes | ✗ |
| user pipeline (Y-floor→Z-edge→plate-band) | within-20 23/46; Y-floor can't replace cy prior; plate-band weak on oblique FS | ✗ (partial ideas kept) |

**Performance:** brute-force SOR was O(n²) (36ms @2k pts, 580ms @8k — a real-time risk);
replaced with voxel-hash SOR (**4.4ms @2k, 46ms @8k**, identical output on all 47 frames).
Region sizes observed: median 1,168 / max 2,023 pts. Target-stage budget ≈ 5ms, well under
the 300ms/frame limit.

**Key design decisions locked:** (a) ROI region kept FULL, narrowing happens in targeting;
(b) cy is a per-config preset (CHS20 wheel proof); (c) OLD detector reused unchanged, only its
input is narrowed; (d) "center-Y" = region's own mid-Y (adapts per frame), not preset cy.

---

## 6. CLPS / CLPS-OK (offload) — study, design, implementation

**Premises (site owner):** ① landed 중엔 컨테이너가 plate를 가려 정확 position 불가 →
3D dev는 landed 중 무의미(상대판정만); ② 하중으로 plate 위치/높이 이동; ③ CST는 lock 없음 →
PLC Landed OFF면 CLPS-OK 인정; ④ LandOK는 3D dev로 보면 틀림 — 별도 로직 필요; ⑤ Landout은
hole-only여도 Landed ON이면 인정; ⑥ CLPS arm = **TWL_Locked**.

**Reverse-replay study** (`roi_offload_replay.py`): real offload recordings all end while
landed (no separation phase), so Mount jobs are replayed **backwards** = offload
(landed→hoist→separation→cone-only). 40 jobs analyzed:
- **VA blindness real:** 3/18 FRONT jobs have no VA detection in the cone-only phase → the
  production instant VA cone-only CLPS-OK rule can **never fire** there; our 3D tracks throughout.
- **landed-phase H = container corner:** `dH_land = H(landed) − H(cone-only) = +22..+169mm`
  (always positive) → after land-off, H **drops** to the plate level within 2–3 frames.
- **plate H settles immediately after separation:** `H_sep vs H_conly ≈ 0mm` (rebound≈0) →
  the post-separation baseline is trustworthy at once.
- **ungated stability rule false-fires while LANDED in 33/40 jobs** → **PLC land-off gate is
  mandatory.**
- production **instant VA cone-only rule fires while LANDED in 4 jobs** (partner cone seen as
  idx1) — a concrete FP against the "CLPS-OK FP intolerable" requirement.

**A/B tuning** (`roi_clpsok_ab.py`, `_replay_cache.csv`): variants (stability + devY thresholds,
dynamic land-off gate, re-land release). Stability-based gated rules fire land-off +2..+5
frames median, **0 fires while landed**. devY<−300 alone lagged on Center configs (max −235
reachable on some) → hole-based conditions demoted to secondary.

**Implemented (C++):**
- **`ChassisRoi.h: ClpsOffload3D`** — arm = landed phase seen && TWL locked && PLC landed OFF.
  **CLPS-OK** = H valid && H ≤ landedH+30 (blocks "hoist paused, chassis dangling") && H stable
  `stableN` frames (≤`stableTol`mm); latched, re-land releases + relearns landedH.
  **CLPS** = H stays ABOVE landedH+30 for `liftN` frames (plate never re-exposed = chassis
  rising with the container). Defaults stableN=12 / stableTol=50 / liftMargin=30 / liftN=15.
- **`_eff_landed`** = PLC `SPRD_Landed` (live) / visual trigger (debug); now drives the VA CLPS
  gate + CST rule. (The PLC bit was parsed-but-unused before.)
- VA CLPS-OK **instant-fire removed** — now landed-OFF-gated + `CLPS_NCOUNT` debounced + released
  on re-land / idx2.
- Combined: `_Out_clps* = VA(gated) || PCA || 3D`. (byte17 bit2 combined-value fix still pending.)
- **res.txt +4 columns** (appended last, prior indices unchanged): `EffLanded;CLPS_3D;CLPSOK_3D;
  LandedH_3D` — the gating bit, the INDIVIDUAL 3D verdicts, the learned baseline. Lets a batch
  run show which path fired per frame.

**Debug harness — `ENABLE_MOUNT_REVERSE=1`** (`[DEBUG]`): each Mount job processed TWICE —
forward → `SAVE_DIR/Mount_Result/`, reversed-as-offload → `SAVE_DIR/Offload_Result/`. Native
Offload jobs → Offload_Result (no reverse). `getAllFiles` now sorts explicitly (was
filesystem-order) so frames are deterministic. Runbook: `CLPS_REVERSE_BATCH_GUIDE.md`.

---

## 7. Known limitations / open items (XT)

1. **Landout / LandOK not reworked yet** — VA landout still has an ungated hole-only path;
   **LandOK does not exist** (premise ④: needs its own logic, not 3D dev). PCA guide-branch
   `cone_pos(−10000)` bug still present.
2. **CLPS/CLPS-OK not validated on REAL separation recordings** — only reverse-replay. Re-verify
   when true TWL-unlock→hoist→cone-only recordings arrive. No real chassis-LIFT data exists, so
   the CLPS (lift) branch is design-only.
3. **byte17 bit2** still set from `PCA_isCLPS_OK` instead of combined — deferred fix.
4. **Stop-position outliers** (`215203`, `150426_451`): chassis not at the preset target →
   ACQUIRING/empty extraction early, uncoverable by single preset key. Data characteristic,
   not a logic bug; frames before the chassis settles are simply not valid targets.
5. **Per-lane live cy** values in process.ini still to be filled; batch FRONT mixes FS+CS so
   config cone-height spreads ~±100mm (live per-lane will be tighter).
6. **VS rebuild + batch smoke-run** pending (no vcxproj change needed).
7. Minor naming: tracker filename prefix `TRK` vs res.txt `ROI_State` value `LOCK` (both =
   LOCKED) — could unify.

---

## 8. Next: IGH

IGH target = **guide top-left closest corner** (different from XT plate corner). Model classes
`igh`(3) + `ighs`(4). Expect: different side-rule / corner definition, a `pc_IGH_detection_naive`
path already exists, and IGH CLPS-OK currently hard-requires both `hole`+`cone` (hole unreliable,
no fallback). The XT ROI→target framework (preset crop → region → narrow → detector →
corner-height) and the `ClpsOffload3D` machine are reusable; the per-type pieces to redefine are
the candidate class filter, the corner geometry, and the preset keys/cone-height.

---

## 9. Center 2-cone end-selection + 3D deviation (2026-07-14, site owner) — IMPLEMENTED (build pending)

**Implemented in C++ (all live-path; needs VS rebuild + batch validation):**
- Flags (`ProcessLogic` ~L9162): `centerTwoCone = Center && chassisLen==40`, `targetMaxX = jobPos=="Big"`,
  `target_on_right = !centerTwoCone && Center && (Big || chassisLen==20)`.
- Cone END-selection helper **`xt_center_end_target(region, targetMaxX, inset)`** (near `igh_target_from_region`):
  SOR → near-Z upper-half tips → outer X extreme (max/min by targetMaxX) → `tx = extreme ∓ inset`;
  end slab (≤230mm) → `ty = median Y`, `tz = 20th-pct Z`. Wired into `computeRoiConePos` XT branch
  AND the FORCE_ROI XT block (used when `centerTwoCone`; single-cone keeps `pc_cone_detection_naive`).
- Hole side (`position_detection_pca`): 2-cone detects the hole on the `targetMaxX` end (matches the cone).
- INI keys `XT_TWOCONE_INSET` (130) and `CORNER_CASTING_W` (167) in `[SETTINGS]`.
- **3D deviation** (`_dev3d_out` / `devFromLatchedRef`): offsets corrected to **X=26, Z=14** (were swapped);
  non-2-cone `dev_x = cone − (tor? hole+26 : hole−26)`, `dev_z = cone − (hole+14)`. 2-cone:
  `dev_x = (cone ±26) − (hole ±167)` (Big +, Small −) — 167 on HOLE (endpoint align), 26 on CONE.
  RS (`cam_pos=="Rear"`) mirrors deviation X and Z (deviation only, positions as-is). Same offset
  swap applied in `deviation_output_pca`.

Below = the original design record.

---

## 9-spec. Center 2-cone end-selection (design record)

**Problem:** on the **center sensor**, a **40ft chassis carrying a 20ft container** shows
**TWO** twist-lock cones in the crop (~430mm apart in X/Gantry). The single-cone corner
detector (`pc_cone_detection_naive`) picks the WRONG cone here (measured **dX ≈ 265mm** off GT
on FRONT/Small) — its internal `target_on_right` filtering is inverted vs the world end that
GT sits at. So the two-cone case needs an EXPLICIT end pick, not the detector.

**Site-owner rule (GT-validated on `groundtruth_0714` + `_0714_2`, 39 XT frames):**
- **Center sensor is used ONLY for SPRD20** (40ft-container / SPRD40 jobs never use center).
- **40ft chassis (center) → 2-cone override:** pick the END by job —
  **Big → max-X cone, Small → min-X cone.** target.X = that outer extreme **∓ INSET ≈ 130mm**
  (cone body half-width), target.Z = near-Z, target.Y = corner height.
- **45ft chassis (center):** cones are far apart → only one in view → **single-cone**, normal
  `target_on_right` (Big → max-X). Not 2-cone.
- **20ft chassis Big (center):** single cone, `target_on_right` (max-X).
- Validation: `target.X = end ∓ 130` reproduces GT — FRONT/Small median |dX/dY/dZ| **5/4/13**
  (25/26 within40), REAR/Big **17/8/3**, ALL **7/5/9**, 95% within40mm. (`tools/` Python
  prototype; the corner detector = 265mm off, replaced by this end pick.)

**Finalized `target_on_right` restructure (ProcessLogic ~L9162, LIVE PLC path):**
```cpp
// SPRD40 never uses center -> every Center job is SPRD20.
bool centerTwoCone   = (cam_pos == "Center" && jobInfo.chassisLen == 40);        // 2-cone override
bool target_on_right = (!centerTwoCone) && (cam_pos == "Center"
                        && (jobInfo.jobPos == "Big" || jobInfo.chassisLen == 20)); // single-cone max-X
```
Only 40ft-center jobs leave the `target_on_right` path (→ 2-cone); 45/Big, 20-chassis, and all
non-center behave exactly as before → minimal regression risk.

**Implementation plan (paired, do together to avoid a regression window):**
1. Helper `xt_center_end_target(region, targetMaxX, inset)` beside `igh_target_from_region`:
   SOR → near-Z upper-half tips → outer X extreme (max/min by `targetMaxX`) → `tx = extreme ∓ inset`;
   from the ≤230mm end slab: `ty = median Y`, `tz = 20th-pct Z` (mirrors the validated Python).
2. Wire into the XT branch of `computeRoiConePos` AND the FORCE_ROI XT refine block: when
   `centerTwoCone`, call the helper with `targetMaxX = (jobPos=="Big")` instead of the detector.
   (Thread `centerTwoCone`/`targetMaxX` in, or derive `cam_pos` from `current_lane_camera`.)
3. `INSET` → `process.ini` key `XT_TWOCONE_INSET` (default 130).

**Deviation sign (for reference, current code):** VA `deviation_output_va` = **hole − cone**
(×5mm/px, Rear inverts X); PCA/3D `deviation_output_pca` = **cone − hole** (+ legacy x/z offset).
The two pipelines are OPPOSITE in sign; VA is the primary PLC output (`hole − cone`).
