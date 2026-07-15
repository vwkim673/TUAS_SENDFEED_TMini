# Assignment — Mark the "correct answer" for chassis/container target finding

## Why
Our system locates the chassis cone (and the container corner-casting) in 3D two ways:
(1) from the 2D AI camera model ("VA"), and (2) from a model-free 3D point-cloud method
("ROI"). We need **your human-marked correct answer** on real jobs so we can measure how
accurate each method is, in different lighting, and improve them.

You are the answer key. Mark where the target REALLY is; we compare the software to you.

---

## Step 1 — Pick the sample jobs
- One chassis type at a time. Start with **XT**.
- For each size config (e.g. **XT 45ft chassis / 40ft spreader**), pick **3–4 real jobs**.
- Spread them across lighting: **sunrise, daytime, sundown, night** (try to get each).
- **MOUNT jobs only** for now (offload is just the reverse — later).
- **Skip** any folder whose name contains **TMINI_LEFT / TMINI_RIGHT** (old site, not ours).

## Step 2 — Run the batch test
Run the current build on the selected jobs. It produces, per frame, a `base_converted.pcd`
(the full 3D scene) plus the software's own results. *(If this is done for you, skip.)*

## Step 3 — For each job, mark the TARGET by hand
The "target" = the **specific cone the crane is working** (and the **container
corner-casting**). If two cones are visible, mark which one is the real target.

**Target corner to use (as seen in the camera/sensor view):**
- **Container corner-casting (hole):** the **lower-left** corner of the casting.
- **XT cone:** the corresponding corner on the **chassis plate** where the casting lands.
- **IGH:** the guide's **top-left** corner.
- **CST:** the **inner guide** corner nearest the casting.
Mark the **same corner the same way every time** — consistency matters most.

For each marked frame, record two things (tools: **CloudCompare**, free):

**(a) Target position** — point-pick the exact corner on `<frame>_base_converted.pcd`,
write down **X, Y, Z in mm** (= Trolley, Height, Gantry).

**(b) Target body** — crop just that cone/casting and save it:
CloudCompare **Segment** tool (scissors / press `T`) → draw a polygon around the object →
right-click to close → press `I` to keep inside → green check → **File ▸ Save ▸ .pcd**.
(Do **not** screenshot — we need the real 3D points. Include the cone, exclude the deck and
the wheel.)

**Which frames to mark:**
- **Chassis cone** — it does **not move** during a job → mark **3 frames**: first clear
  frame, one in the middle, and the **last frame before the container covers it**. Note the
  frame where the container starts to cover the cone.
- **Container casting** — it moves **down** → mark **every ~5th frame** until it lands.

## Step 4 — Compare to the software (the key output)
For each marked frame, open these together in CloudCompare and compare **three points**:
- **GT** — your marked point.
- **ROI (software 3D method)** — `<frame>_ROI_TRK_conePos.pcd`.
- **VA (2D model)** — `<frame>_AB_va_conePos.pcd`.

Write down, per frame, **how far the ROI point and the VA point each are from YOUR point**,
in T, H, G (mm). Flag any frame where the camera/VA looks wrong but ROI still matches you
(or vice-versa), and note the lighting.

## Step 5 — Deliver
- **One spreadsheet (CSV)**, one row per (job, frame, object):
  `job, frame, sensor, object, is_target, visibility, lighting,
   GT_T, GT_H, GT_G, ROI_T, ROI_H, ROI_G, VA_T, VA_H, VA_G, notes`
  (`frame` = the image file name **without extension**; `visibility` = full/partial/
  occluded/absent; `lighting` = sunrise/day/sundown/night.)
- **The saved target-body PCDs**, named `<frame>_GT_coneBody.pcd` (and
  `<frame>_GT_holeBody.pcd` for the casting).

## Tips
- Start with **one full job**, send it back, and wait for an OK before doing the rest — so
  we confirm the corner choice and the file format are right.
- Detailed reference (corner definitions, file list, coordinate notes):
  `GROUNDTRUTH_GUIDELINE.md`.
