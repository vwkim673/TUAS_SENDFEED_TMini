# Ground-Truth Labeling Guideline — TUAS South Endfeed ROI Validation

Purpose: produce human ground truth so we can measure how well the system finds the
**chassis target** and the **container corner-casting (hole)** — both in 2D (the detection
box) and in 3D (the target position) — and improve it.

You are creating the "answer key". The software's job is to reproduce what you mark.

---

## 0. The one fact that makes this easy

Each frame's **image is 512 × 424 pixels, and the 3D point cloud is the same 512 × 424
grid**. A pixel you click on the image corresponds 1-to-1 to a 3D point:

```
3D point index = (pixel_y * 512) + pixel_x
```

**So for most labels you only need to draw on the 2D image** — we convert your pixels to
3D (T, H, G millimetres) automatically. You only open the 3D cloud for the small
"gold-standard" subset in Section 5.

---

## 1. What to label, per object

For every frame you label, mark each **target object that is visible**:

| Object | What it is | Mark it for |
|---|---|---|
| **hole** | the container **corner-casting** (the twist-lock hole block on the container bottom) | every job (it's the container side) |
| **cone** | XT chassis twist-lock **cone**, *unloaded* (no container on it) | XT jobs |
| **land** | XT chassis corner *loaded* (container casting resting on the chassis) | XT jobs, after landing |
| **igh / ighs** | IGH chassis guide cone / guide-surround | IGH jobs |
| **cstf/csts/cstb/cstsl** | CST chassis guide pieces | CST jobs |

For each visible object you record **two things**:
1. **The box** (2D ROI) — a rectangle tightly around the object.
2. **The target point** — ONE pixel, the exact reference corner (see Section 3).

You also flag **which object is the job's target** (`is_target = 1`) — there can be more
than one cone/casting in view; only one is the one the crane is working.

---

## 2. Which image to annotate

Use the clean image in each job's **`Image/`** folder, e.g.
`...\<job>\Image\20260126_110914_638_TMini_FS1_Image.jpg` (512 × 424).
**Do NOT** use the `..._debug_result.jpg` (it has the software's own coloured boxes drawn
on it — that would bias you). It's fine to *glance* at the debug image to understand the
scene, but mark on the clean one.

Pixel origin is **top-left**: x increases to the right (0–511), y increases downward
(0–423).

Always record the exact frame name **without extension**, e.g.
`20260126_110914_638_TMini_FS1_Image` — this is how your labels get matched to the
software's output.

---

## 3. Target-point definitions (THE important part)

The "target point" is the single reference corner the crane aligns to. Mark it as
precisely as you can — this is what we measure against. All directions are **as seen in
the image** (sensor view).

- **hole (container corner-casting):** the **lower-left corner** of the casting block.
- **XT cone (unloaded):** the point on the chassis **plate** where the casting's
  lower-left corner should sit when correctly landed (i.e. the matching corner on the
  chassis cone/plate).
- **XT land (loaded):** same corner, but on the landed casting/chassis interface.
- **IGH guide:** the **top-left closest corner** of the guide.
- **CST guide:** the guide reference corner (CST has the most clearance — pick the inner
  guide corner nearest the casting).

If you are unsure which corner, mark the one a person would line the container up to, and
add a note. Consistency matters more than perfection — mark the **same** corner the same
way every time.

---

## 4. Required labels per frame (Tier 1 — 2D, do this for all sampled frames)

One row per (frame, object). Columns:

```
job, frame, sensor, object, is_target, visibility,
box_x1, box_y1, box_x2, box_y2, tgt_px_x, tgt_px_y, state, notes
```

- **job** — job folder name.
- **frame** — image name without extension (the matching key).
- **sensor** — FS (front), CS (centre), RS (rear) — from the job/file name.
- **object** — one of the class names above.
- **is_target** — `1` if this is the object the crane is working this job, else `0`.
- **visibility** — `full` / `partial` / `occluded` / `absent`.
  - `occluded` = the object is there but hidden (e.g. the descending container covers the
    cone). Still mark the box where you believe it is, and set visibility=occluded.
  - `absent` = genuinely not in this frame (don't draw a box; record the row with absent so
    we know it was checked).
- **box_x1..y2** — top-left and bottom-right pixel corners of the tight box.
- **tgt_px_x, tgt_px_y** — the target-point pixel (Section 3). Leave blank if absent.
- **state** — for the chassis object: `unloaded` / `loaded`. For the job moment:
  add `landed` if the container is resting, `airborne` if still hanging. (Free text ok.)
- **notes** — anything odd: flapping, two cones visible, glare, partial cloud, etc.

### Sampling — how many frames

- **Chassis target (cone / land / guide):** the chassis does **not move** during a job, so
  you do **not** need every frame. Label it on **3 frames**: the first clear frame, one in
  the middle, and the **last frame before the container starts to occlude it**. Add the
  frame number where occlusion begins (notes).
- **Container corner-casting (hole):** it **moves down** through the job. Label it on
  **every ~5th frame** from when it appears until it lands.
- **Always also label** any frame where: the cone changes unloaded→loaded, the detection
  looks wrong, two candidate cones are visible, or the chassis appears to shift.
- If a job is short (<~30 frames), just label every frame.

---

## 5. Gold-standard 3D target (Tier 2 — subset, ~3–5 frames per job)

For a **few** frames per job (at least one clean chassis frame + one clean hole frame),
also mark the target **directly in 3D** for highest accuracy. Use **CloudCompare** (free):

1. Open the frame's full cloud: `...\<job>\<frame>\<frame>_base_converted.pcd`.
2. Use the **point-picking** tool, click the exact target corner (Section 3).
3. Record the point's **X, Y, Z** — these are **T (trolley), H (height), G (gantry)** in mm.

Add these to a second sheet:

```
job, frame, object, tgt3d_T, tgt3d_H, tgt3d_G, notes
```

This is the most accurate target and is what we calibrate the 2D→3D conversion against.
(Optional, advanced: if you want to also bound the object in 3D, use CloudCompare's
**Segment / crop box** around the target object and note the box min/max T,H,G. Not
required — the 2D box already gives us the 3D region via the grid mapping.)

---

## 5b. Saving the target BODY (cropped 3D) in CloudCompare

To save the target object's 3D points (the "target body / crop"):

1. Open `...\<job>\<frame>\<frame>_base_converted.pcd` in CloudCompare.
2. Select the cloud in the tree, then click the **Segment** tool (scissors icon, or
   press **`T`**).
3. **Left-click** repeatedly to draw a polygon around the target object; **right-click**
   to close the polygon.
4. Press **`I`** to keep the points **inside** the polygon (the "segment in" button).
5. Click the green **check** to confirm. CloudCompare makes a new cloud named
   `..._base_converted.segmented`.
6. Select that new cloud → **File ▸ Save** → choose **Point Cloud (*.pcd)**.

Notes:
- Do NOT use a screenshot / the snipping tool — we need the actual 3D points, not an
  image. The Segment tool above produces a real 3D sub-cloud.
- Save as **.pcd** (binary or ASCII both fine), **XYZ only** (if asked about scalar
  fields, none are needed).
- The crop is automatically in the correct coordinate frame (same as
  `base_converted`), so it lines up with the software output with no conversion.
- A rough but tight polygon is fine — include the cone/casting, exclude the surrounding
  deck and the wheel.

### File naming for the gold-standard outputs (so they match automatically)

Save into the same `<frame>` folder, named after the frame:
- target point: put it in the Tier-2 CSV (T,H,G). Optionally also save the single picked
  point as `<frame>_GT_conePos.pcd` (for hole: `<frame>_GT_holePos.pcd`).
- target body: `<frame>_GT_coneBody.pcd` (chassis) and/or `<frame>_GT_holeBody.pcd`
  (container casting).

## 5c. Comparing against the software output (per frame, in CloudCompare)

The batch run (FORCE_ROI mode) already writes these into each `<frame>` folder. Load them
together with your ground truth to compare visually + by the numbers:

| File | What it is |
|---|---|
| `<frame>_base_converted.pcd` | full scene (context) |
| `<frame>_ROI_TRK_roiCrop.pcd` | the ROI crop the software searched (locked frames; `ROI_ACQ` before lock) |
| `<frame>_ROI_TRK_coneSlab.pcd` | the nearest-Z slab it used as the cone |
| `<frame>_ROI_TRK_conePos.pcd` | **software's cone target point** |
| `<frame>_AB_va_body.pcd` | the 2D-VA detection box mapped to 3D |
| `<frame>_AB_va_conePos.pcd` | **VA's cone target point** |
| `<frame>_GT_coneBody.pcd` / `_GT_conePos.pcd` | **your ground truth** |

Then for each frame compare three points: **GT (yours) vs ROI (software) vs VA (2D)**.
The job's `..._res.txt` also already lists, per frame: `ROI_T/H/G` (software cone),
`AB_dT/dH/dG` (software-minus-VA), and the tracker `ROI_State`. Record, per frame, the
distance from the software's `conePos` to **your GT point** (T, H, G separately) — that
is the number we ultimately care about.

## 6. Tools

- **2D boxes + points:** CVAT (cvat.ai) or Label Studio — both free, support rectangles,
  points, per-object attributes (is_target, visibility, state), and CSV/JSON export. Set
  up object labels = the class names above, plus a "target-point" point label.
  - If a full tool is too heavy, any image viewer that shows the **cursor pixel
    coordinate** (e.g. GIMP, IrfanView) works — read off the box corners and the target
    pixel and type them into a spreadsheet. The images are small, this is quick.
- **3D point pick:** CloudCompare (point-picking tool shows XYZ).

Whatever tool, **deliver the data in the two CSV schemas above** (Tier 1 boxes, Tier 2
3D points). One pair of CSVs total (all jobs) is fine — keep the `job` and `frame` columns
filled so rows can be traced.

---

## 7. Worked example (one frame, XT Mount, container descending)

Frame `20260325_180248_125_TMini_FS4_Image`, front sensor, two cones visible, crane on the
right-hand one, container half-landed:

Tier 1 rows:
```
job, frame, sensor, object, is_target, visibility, box_x1,box_y1,box_x2,box_y2, tgt_px_x,tgt_px_y, state, notes
<job>, 20260325_180248_125_TMini_FS4_Image, FS, cone, 1, full, 300,210,360,275, 312,268, unloaded, "target cone (right)"
<job>, 20260325_180248_125_TMini_FS4_Image, FS, cone, 0, full, 120,205,178,268, 130,262, unloaded, "other cone, not target"
<job>, 20260325_180248_125_TMini_FS4_Image, FS, hole, 1, partial, 295,150,365,215, 300,210, airborne, "casting descending toward target cone"
```

Tier 2 row (same frame, picked in CloudCompare):
```
job, frame, object, tgt3d_T, tgt3d_H, tgt3d_G, notes
<job>, 20260325_180248_125_TMini_FS4_Image, cone, 835, -240, 1815, "target cone corner"
```

---

## 8. Quick checklist per job

- [ ] Chassis target labelled on 3 frames (first clear / middle / last-before-occlusion).
- [ ] Container casting labelled every ~5th frame through the descent.
- [ ] `is_target` set correctly when more than one cone/casting is visible.
- [ ] Occlusion-start frame noted.
- [ ] At least 1 chassis + 1 hole frame given a CloudCompare 3D point.
- [ ] Frame names recorded exactly (no extension), job name filled.
- [ ] Anything ambiguous written in `notes`.

Skip the `*_TMINI_LEFT` / `*_TMINI_RIGHT` job folders — those are old data from another
site and are not part of this validation.
