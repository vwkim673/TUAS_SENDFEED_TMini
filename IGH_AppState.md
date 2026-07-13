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

## 7. Remaining / next steps
1. **C++ batch validation** — rebuild, run IGH batch (`FORCE_ROI_MODE=1`, DEBUG_BATCH), then a C++
   vs GT compare script → dump to `Comparison\IGH_cpp\` (write once batch-output paths are known).
2. **Merged CS_Big_CHS40 offset** — site owner: whole-guide target point (whole + what offset).
3. **Dedicated IGH INI keys** — lift the hardcoded IGH crop (X/Y/Z, coneY, xspan) out of `fillRoiParams`.
   Note the preset-key scheme lumps IGH Big CHS40/45 into one 4XF key (why the crop is hardcoded).
4. **CS dH −16** — if it matters, inspect the guide-top override (`denseExtreme` may back off to a
   lower dense level when the top has <10 pts in a 40mm window).
5. CST not started.
