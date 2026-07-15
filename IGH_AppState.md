# IGH Application State — target positioning (2026-07)

Handoff snapshot. **XT released as v1.13 (committed 8794b80).** IGH now **ported to C++**
(`ightop` rule) and validated offline vs GT; **C++ batch validation still pending.** Companion:
`XT_AppState.md`, `ROI_EXTRACTION_PIPELINE.md`, `V1.13_DEPLOYMENT_REPORT.md`.

Axes: **X = Gantry (lateral), Y = Height (up +), Z = Trolley (depth, near-Z = toward sensor)**.
GT `groundtruth.txt` cols labelled `T;H;G` are positionally **(Gantry, Height, Trolley) = (X,Y,Z)**
= the app's pcl x/y/z directly.

---

## 1. Domain facts (site owner)
- **IGH = XT chassis + a GUIDE**; target = the guide's **top-left closest corner** (guide TOP,
  ~90mm ABOVE the XT plate corner).
- Guide: ~200mm wide X (single), TILTED Y-Z (top nearest-Z), ~200mm tall Y, ~100mm deep Z; plate
  adds ~200mm Z behind.
- **Two guides**: SEPARATED (gap) or **MERGED** (40chs+20guide joined, ~**480mm** X).
  - SEPARATED: Big = one side (side-rule). MERGED (**CS_Big_CHS40**): **target the WHOLE**; GT labels
    one guide arbitrarily → merged NOT scored vs GT; **whole-target OFFSET still TBD (site owner)**.

## 2. Data
- GT: `GroundTruth\IGH\groundtruth.txt` (states CONLY/LANDING/LANDED; per-job 3 frames).
- Clouds: `OriginalData\IGH\<FRONT|REAR>\BatchResult\Mount_Result\<job>\<frame>\<frame>_base_converted.pcd`.
- All jobs Mount/Big/SPRD20/CHS40|45. Configs: **FS_Big_CHS40/45** (FRONT), **CS_Big_CHS40/45** (REAR).

## 3. Chosen rule = `ightop` (NOT ighface)
`ightop` = **XT near-face corner (x,z) + Y override to GUIDE TOP** (dense-max Y at corner column).
Beat `ighface` on the 3 singles AND simpler/closer to the C++ XT chain. `roi_igh_eval.py --rule ightop`.

## 4. C++ port (DONE this session — uncommitted until this checkpoint)
- **`igh_target_from_region(region, tor)`** in `SampleVisionaryTMini.cpp` (before `computeRoiConePos`):
  `chassis_roi_sor → chassis_roi_near_face(80) → pc_cone_detection_naive → sparse_edge_rescue`
  (= `new_target`, gives x,z) then **Y = `denseExtreme(corner-column Y, 40,10, findMax=true)`** = guide top.
  `chassis_roi_near_face(...,upperHalf=true)` confirmed == Python `new_target` upper-half+near-Z.
- **Wired**: `computeRoiConePos` IGH branch → `igh_target_from_region`; **FORCE_ROI block IGH refine**
  (overrides `_cone_pos`, saves `_Region.pcd` + `_ROI_target.pcd`).
- **IGH crop SEPARATED from XT** in `fillRoiParams` (own X/Y/Z, NOT shared HEIGHT_MMR/TROLLEY_MMR or
  the colliding IGH_4XF RNG key). Sized from GT spread — **hardcoded, TODO: dedicated IGH INI keys**:
  | | FS | CS (REAR) |
  |---|---|---|
  | X (Gantry) | 500..1150 | 350..1300 (merged whole) |
  | Y (Height) | −600..100 | −600..100 |
  | Z (Trolley) | 1400..2050 | **1150**..2050 (near-Z front incl.) |
  | coneY (guide top) | −4/−7 | −77/−66 |
  | xspan | 300 | CHS40=**600** (merged) |
- Non-ROI (pure VA) IGH still uses `pc_IGH_detection_naive` (no regression).

## 5. Python tooling updates (`tools/roi_igh_eval.py`)
- `--rule ightop`; dumps `_presetCrop / _Region / _tgt / _gt / _GTroi` to `Comparison\IGH_target\`
  (`_Region` capitalised to match the user's compare tool + the C++ output name).
- **`IGH_XSPAN={'CS_Big_CHS40':600}`** — merged region was capped at 300mm; cx is merged centre,
  so widen xspan → region now covers full ~460–535mm.
- **`IGH_GANTRY_HALF={'CS_Big_CHS40':450}`** — wider crop for merged.
- **`R.TROLLEY_BACK=500` (was 250), `R.HEIGHT_MAX=100` (was 50)** — CS jobs have ~600mm Z spread and
  targets to Y~56; the default fixed margins dropped near-Z fronts (214751, 000020) and high targets.
  **Root cause of "GT outside preset": preset centre = GT mean, but fixed margins < actual spread.**
- **`moved_jobs()` exclusion** (`--keepmoved` to disable): chassis MOVES to job position after the
  container lowers in a FEW jobs → CONLY frame is at the START position → preset misses. Excluded 5
  frames (`190047`, `203549`, `043256`). Most jobs stationary (|dX| med 5–11mm).

## 6. Results (`ightop`, CONLY, non-moved, after crop widening)
| config | n | dT / dH / dG | within20 | status |
|---|---|---|---|---|
| FS_Big_CHS40 | 10 | 3 / 0 / 7 | 6/10 | ✅ |
| FS_Big_CHS45 | 12 | 10 / 5 / 11 | 10/12 | ✅ |
| CS_Big_CHS45 | 11 | 5 / −16 / 5 | 6/11 | ✅ (dH −16) |
| CS_Big_CHS40 | 10 | −387 (whole) | — | merged, offset TBD |
- overall within20 18→**24/43** after crop widening. Guide-bottom mis-catch: **resolved** by the
  crop widening (user-confirmed). CS dH −16 residual remains (guide-top read ~16mm low) — minor.

## 7. IGH INI keys — DONE (2026-07-14)
The hardcoded IGH crop is now driven by `process.ini` **`IGH_ROI_40` / `IGH_ROI_45`** keys under
`[PRESET_FRONT_3D]` / `[PRESET_REAR_3D]` (VA-independent backup, same role as XT's ROI crop):
- Schema: `xMax;xMin;yMax;yMin;zMax;zMin;coneY;xspan` (8 fields). Keyed by CHS length directly so
  Big CHS40/45 (which share one preset key, and differ in guide-top coneY + merged xspan) are
  separable — the reason the crop was hardcoded before.
- C++: `PRESET_POS::IGH_ROI_BOX` (defaults == the old hardcode → identical behaviour if keys
  absent); throw-safe `parseIghRoi` in `parseProcessINI`; `fillRoiParams` IGH branch now selects
  the box by `CURRENT_SENSOR_POSITION` + `chassisLen`.
- Written into `build/INI/process.ini` and `build/Release/INI/process.ini`.
- Header-only + existing cpp → **no vcxproj change; needs a VS rebuild**.

## 8. C++ batch validation — FRONT DONE (2026-07-14)
Ran the batch with **`FORCE_ROI_MODE=1`** (res.txt gains `ROI_State;ROI_T;ROI_H;ROI_G;...`;
`ROI_T/H/G = _roi_detected.x/y/z = igh_target` for IGH). **`tools/roi_igh_cpp_eval.py`** reads
those columns per job (NOT pcd files — more robust), joins to GT, prints the roi_igh_eval sheet.

FRONT **CONLY** (chassis-only = correct ref state) — C++ reproduces the Python offline result:
| config | C++ dT/dH/dG (mm) | within20 | Python §6 |
|---|---|---|---|
| FS_Big_CHS40 | 5 / −1 / 3 | 7/10 | 3/0/7 |
| FS_Big_CHS45 | 12 / 3 / 7 | 8/12 | 10/5/11 |
- ALL states: dT/dG stay tight (4–8mm) but **dH → +53..58** in LANDED/LANDING — expected, NOT a
  bug: the container casting rests ON the guide top and occludes the GT-labelled guide-top height
  (XT's "landed H reads the container corner" analog). CONLY is the height-valid state.
- **Port confirmed faithful.** CSV: `Comparison\IGH_cpp\front_conly.csv`.

REAR **CONLY** (FORCE_ROI re-run 2026-07-14 11:02) — also reproduces Python:
| config | C++ dT/dH/dG (mm) | within20 | Python §6 | note |
|---|---|---|---|---|
| CS_Big_CHS45 | 3 / −19 / 4 | 6/10 | 5 / −16 / 5 | ✅ (dH −19 = known guide-top-read-low residual) |
| CS_Big_CHS40 | **−355** / −0 / 10 | 1/9 | −387 (merged) | ⚠️ merged, NOT a bug — see below |
- **CS_Big_CHS40 dT −355 is EXPECTED**: merged config (40chs+20guide ~480mm), C++ targets the
  WHOLE-guide corner while GT labels one arbitrary joined guide → not a valid single-corner ref.
  C++ −355 ≈ Python −387 = same behaviour reproduced. Whole-target offset still TBD (site owner).
- ALL-state dH/dG inflation same as FRONT (landed container occludes GT height). CSV:
  `Comparison\IGH_cpp\rear_conly.csv`. (2 CONLY frames ROI-sentinel, 3 moved excluded.)

**Port validated on BOTH sensors** for every single-guide config (FS_40, FS_45, CS_45) to a few
mm, matching the Python reference. Only open target case = the merged CS_Big_CHS40 offset.

## 9. Unified end-selection rule + IGH GT mislabel (2026-07-14, site owner)
New GT sets `groundtruth_0714` (XT) and `groundtruth_0714_2` (XT+IGH), Center-sensor XT 40ft/20
and REAR IGH merged, where the ROI deliberately contains BOTH cones/guides. Site owner fixed the
targeting DIRECTION — **the same for XT and IGH**:
- **REAR / Big job → target the max-X (+X) END** of the structure.
- **FRONT / Small job → target the min-X (−X) END.**
- Target = that end's near-Z corner, offset a small **per-type inset** from the extreme:
  **XT ≈ 127mm** (cone → corner-casting), **IGH ≈ 22mm** (guide corner, ~at the extreme). Z = near-Z,
  Y = corner height. inset should become a per-type `process.ini` key.

Validation (Python prototype, `tools/`): XT vs REAL GT — FRONT/Small (min+127) median |dX/dY/dZ|
**9/4/12**, 4/4 within20; REAR/Big (max−127) **11/8/3**, 4/6 within20. So the rule reproduces XT GT
to ~10mm.

**IGH GT was labelled on the WRONG (opposite) end** — the worker marked the min-X guide, but for
REAR/Big the rule wants the max-X (40-guide) end (confirmed on top-views: GT 'G' sits at the min-X
extreme while the guide extends to max-X). Reflecting the mislabelled GT to the max end, the rule's
max-X corner reproduces it to dX 9 / dZ 7 (dY 53 = mirror keeps the min-end height; refine later).
→ **This RESOLVES the old "merged CS_Big_CHS40 whole-target offset TBD"**: it is NOT a whole/centroid
target — it is the job-appropriate END corner (Big→max-X), same rule as XT. **IGH GT needs re-labelling
to the max-X end** before it can be scored directly.

Note the earlier §3 `ightop`/`guide top-LEFT` framing is superseded for the merged/Big case: the end
is chosen by job side (Big→+X), not fixed to the left corner.

### Remaining / next steps
1. **IGH GT re-label** — worker to mark the max-X (Big/40-guide) end, not the min-X end. Then score
   IGH directly against the unified rule (no mirroring).
2. **Port the unified end-selection rule to C++** — XT Center 2-cone + IGH: `end=(REAR||Big)?maxX:minX`,
   `target.X = end ∓ inset[type]`, Z=near-Z, Y=corner. inset per-type INI key (XT 127 / IGH 22).
   **(Doing XT first — see below.)**
3. **CS dH −16/−19** — guide-top override refinement (`denseExtreme` may back off when the top has
   <10 pts in a 40mm window).
4. **VS rebuild** — the IGH_ROI INI refactor (§7) + any new rule still need a Visual Studio rebuild
   to ship.
5. CST not started.

**STATUS 2026-07-14: IGH paused here. Switching to XT-only** (implement/validate the unified
end-selection rule for the XT Center 2-cone case first; return to IGH after its GT is re-labelled).
