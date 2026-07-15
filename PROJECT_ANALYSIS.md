==============================================================================
 TUAS SOUTH ENDFEED — PROJECT ANALYSIS & REVIEW NOTES
 Target: SampleVisionaryTMini  |  Branch: v1.2  |  program_version = "1.2"
 Generated for review (analysis only — no source changes made)
==============================================================================


###############################################################################
# 1. WHAT THIS PROJECT IS
###############################################################################

Started as SICK's `sick_visionary_cpp_samples` (demo code for Visionary-T Mini
ToF cameras). The SampleVisionaryTMini target has been turned into a production
industrial vision application — the "TUAS South Endfeed App".

It guides an automated container-terminal crane spreader placing/removing
shipping containers onto truck CHASSIS, using a downward-looking ToF camera per
lane. It fuses 2D object detection (YOLO/TensorRT) with 3D point-cloud geometry
(PCL/Open3D, PCA) to produce four outputs, reported to a PLC over a raw TCP byte
protocol:

  1. Landout / Land-OK detection      (mount cycle)
  2. CLPS  - Chassis Lift Prevention  (offload cycle)
  3. CLPS-OK detection
  4. 3D deviation between container and chassis (mm)

Other targets (SampleVisionaryS/T/TVGA, SampleAutoIP) and sick_visionary_cpp_shared
(the camera SDK: CoLa 2 control, data streaming, point-cloud generation) are
UNMODIFIED SICK scaffolding. Only SampleVisionaryTMini/ contains the custom app.


CHASSIS TYPES
-------------
  XT   - external type, cone target.          Model classes: cone(1) + land(2)
  IGH  - internal, cone AND guide around.      Model classes: igh(3) + ighs(4)
  CST  - internal, guide only.                 Classes: cstf(5)/csts(6)/cstb(7)/cstsl(8)

Model classes (9): hole(0), cone(1), land(2), igh(3), ighs(4), cstf(5),
                   csts(6), cstb(7), cstsl(8)
  hole(0) = container corner-casting / twist-lock hole.

Target appearance varies by camera position (Front/Center/Rear), job position
(Big/Small/Center), and chassis length (20/40/45 ft) — handled via the
PRESET_POS fallback table.


###############################################################################
# 2. ARCHITECTURE  (SampleVisionaryTMini/)
###############################################################################

SampleVisionaryTMini.cpp  — ~10k-line single translation unit. Heart is
  ProcessLogic() which runs two pipelines and OR-combines them:
    VA  (2D image): chassis_type_selection_va -> target_selections_va ->
                    deviation_output_va -> landOut_detected_va / clps_detection_va
    PCA (3D cloud): target_pointcloud_extraction[_preset] -> position_detection_pca
                    -> deviation_output_pca (+ CLPS_Detection_PCA)
    Combined: bLandOut/bClps/bClpsOk = VA || PCA, with N-count debouncing
              (LDO_NCOUNT, CLPS_NCOUNT, per-pipeline _count accumulators)

Inference backends (3 present, 1 active):
  - TrtRunner (TensorRT, .engine)  -> ACTIVE backend
        model/20251201_TUAS_SEndfeed_3DSP.engine
        YOLOv8-style output [1,C,8400] -> decodeYoloV8_1xCxN + NMS
  - VA (ONNX Runtime + CUDA)        -> present, call site commented out
  - YOLO11Detector (YOLO11.hpp)     -> present, commented out

Threads (main, non-debug):
  - socketClient            : PLC TCP comms
  - thread_tmini_control    : camera connect/setup
  - thread_tmini_data_stream: frame grab -> triple-buffered DataStack
  - processingThread        : runs ProcessLogic
  - data_save_thread        : async PLY/img/log save (ThreadSafeQueue)
  - logWriterThread / jobLogWriterThread : async log files

PLC protocol:
  - parseCommand() decodes 50-byte input (heartbeat, crane/lane ID, mount/offload
    cycle, job pos, spreader twistlock/landed, chassis type & length, commands)
  - makeCommand() encodes 50-byte output (positions, deviations, detection flags)

Config: external INI via miniINI.h (parseINI/parseProcessINI/parseIPINI) —
  thresholds, presets, per-lane camera IPs (Front/Center/Rear), tilt, sensor
  height. INI + model files are RUNTIME ASSETS, not in the source tree.

Modes (INI flags): normal (PLC), MODE_DEBUG, DEBUG_WITH_FILES, DEBUG_BATCH_JOB
  (OfflineDebugBatchProcessingThread replays saved jobs -> *_res.txt).

Deps: Windows, CUDA, TensorRT, ONNX Runtime, OpenCV, PCL, Open3D, Boost.


###############################################################################
# 3. APPLICATION FLOW  (main processing stage step-by-step)
###############################################################################

STARTUP  (main, line ~9762)
  1. Single-instance mutex (by exe path) -> parseAppName.
  2. Start logWriterThread; install Ctrl-C handler + minidump filter.
  3. parseINI / parseProcessINI / parseIPINI (thresholds, presets, camera IPs).
  4. Load TensorRT engine, warm-up infer on initialize.jpg, set
     model_loaded / model_initialized.
  5. Launch threads: socketClient, thread_tmini_control,
     thread_tmini_data_stream, processingThread, data_save_thread.

PLC COMMS LOOP  (socketClient, ~9690)
  connect -> loop:  recv 50B -> parseCommand -> sleep 100ms -> makeCommand
                    -> send 50B.  On disconnect: reset vars, wait 2s, reconnect.
  - NewJobPulse EDGE triggers: reset_jobVariables -> selectTargetSensor ->
    build job folder name -> resolve JobInfo + preset box/range from PRESET_POS
    by (position x chassis x length x jobpos).
  - makeCommand encodes current_process_results back to PLC.

CAMERA GRAB  (tmini_data_stream, ~6261)
  Poll ~150ms: stepAcquisition -> getNextFrame -> build intensity cv::Mat
  (16->8-bit->BGR) + generate/transform point cloud -> makePCL_PointCloud ->
  dataStack.Update_Stack (triple-buffer, keep-latest). Optional live imshow.

MAIN PROCESSING STAGE  (processingThread ~9054 -> ProcessLogic ~8732)
  Per job: open res.txt, write header. Then loop every 200ms:
  1. dataStack.Stack_Ready() -> Retrieve_Stack(img, cloud, ts).
  2. Save triggers: save_trigger_by_landed / _TWL_Locked (set in parseCommand on
     spreader edges) -> push SaveTask (job-log); else if enable_logging -> push
     raw frame.
  3. makePCL_PointCloud -> pointCloud_tilt_transform(TILT_ANGLE_DEG).
  4. ProcessLogic — the core:
       - clone image; compute cam_pos (Front/Center/Rear), target_on_right.
       - trt_inference -> det_sorted_objects[class][...] (2D boxes).
       - chassis_type_selection_va (XT/IGH/CST by class counts).
       - target_selections_va -> pick target_hole / target_cone / target_guide
         per camera/jobpos/length; draw boxes.
       - pre_land_chassis_position_va; debug_landed_trigger (landed flag).
       - deviation_output_va -> 2D dev X/Y (px & mm).
       - mount  -> landOut_detected_va
         offload -> clps_detection_va (CLPS / CLPS-OK)
       - Preset fallback if hole/cone missing (uses JobInfo.presetBox).
       - target_pointcloud_extraction[_preset] -> crop hole/cone/guide sub-clouds.
       - position_detection_pca -> 3D positions via PCA + clustering.
       - deviation_output_pca -> 3D dev X/Y/Z + landout/clps/clpsok (PCA).
       - Combine: bLandOut/bClps/bClpsOk = VA || PCA; fill ProcessResults.
  5. Append log_lines to res.txt.

OUTPUT SIDE
  - data_save_thread (~6023): blocking tsq.pop -> save_to_drive (jpg +
    compressed PCD), with a 5-item batch-drain.
  - logWriterThread / jobLogWriterThread: queue-backed async log files.


WHAT'S BUILT vs. CONSPICUOUSLY NOT
----------------------------------
Built & working: TRT 2D pipeline, chassis-type/target selection, VA + PCA
  deviation, landout (both pipelines), preset fallback, triple-buffer grab,
  async save/log, PLC encode/decode, job/preset routing, reconnect.

Disabled / stubbed (candidates for "to add"):
  - ONNX (VA) and YOLO11 paths commented out.
  - start_server path unused (client mode only).
  - CLPS / CLPS-OK effectively non-firing across the dataset (see Section 5) —
    gating likely incomplete.
  - No socket watchdog/timeout.
  - Live-log header doesn't match the columns actually written (see Section 4).


###############################################################################
# 4. ISSUES  (logging / grabbing / memory / network)
###############################################################################

DATA LOGGING
  L1. Live res.txt HEADER != DATA COLUMNS. Header in processingThread (~9077)
      ends at "...SEP_DEV_Y" and starts with "Filename", but rows START with a
      numeric frame_index (~9111) and ProcessLogic appends two extra trailing
      fields CLPS_OUT;CLPS_OK_OUT (~8974). So live logs are off-by-2 columns and
      mislabel column 1. (The BATCH header is the correct 51-col one — two header
      definitions have diverged.)
  L2. res.txt is reopened/closed EVERY frame (ios::app, ~9180-9189): open+write+
      flush+close per frame on the processing thread; slow disk stalls processing.
  L3. logMessage does a printf to stdout on EVERY call (~1156) and the file writer
      flushes with std::endl per line — heavy console/IO in the hot path; plus
      per-frame TRT/printf debug spam.
  L4. Log writer holds queueMutex while writing to disk (~1078-1082). Every thread
      that logs blocks on that mutex during disk I/O — slow disk serializes all
      threads.

DATA GRABBING
  G1. Polling, not streaming: stepAcquisition+getNextFrame gated ~150ms (~6 fps);
      processing polls 200ms (~5 fps). Mismatch silently drops frames (acceptable
      "latest-only", but the throttle is hard-coded, not configurable).
  G2. Dead dedup guard: last_ts is re-declared INSIDE the loop (~9118) so
      "last_ts < ts" is always true — the "skip if not newer" check does nothing.
      Masked only because DataStack's isUsed flags dedup. Misleading/fragile.
  G3. No fault escalation on grab failure: repeated getNextFrame/stepAcquisition
      failures just log + sleep(30ms) and spin; sensor_fault is never set, so the
      PLC isn't told the feed is dead.

MEMORY
  M1. tsq save queue is UNBOUNDED (ThreadSafeQueue, ~779). PCD compression is
      CPU/IO-heavy; if save throughput < push rate during continuous logging, the
      queue grows without limit (each task ~ 2 cv::Mat + a full cloud) -> steady
      memory growth / eventual OOM. No max-size or drop-oldest policy.
  M2. Unsynchronized cross-thread struct access (concurrency, not a leak but
      serious): current_process_results is WRITTEN by processingThread
      (ProcessLogic) and READ by makeCommand on the socket thread with NO mutex
      (~1985+, ~9178). Multi-field reads can tear -> the PLC can receive
      inconsistent position/deviation values from two different frames. Many other
      shared globals (job info, flags) are likewise lock-free.

NETWORK TIMING
  N1. No socket timeouts. recv blocks indefinitely (~9726). On a half-open TCP
      (PLC powers off / cable pull without FIN), the thread hangs forever — no
      heartbeat, no reconnect, outputs go stale. Needs SO_RCVTIMEO + watchdog.
  N2. TCP framing assumption. parseCommand assumes each recv yields exactly one
      50-byte message at buffer[0..49]. TCP is a byte stream — a short read
      (partial frame) or coalesced reads will misparse. No length-prefix /
      accumulate-to-50 logic. Latent corruption bug.
  N3. Fixed 100ms sleep between recv->send (~9736) caps the PLC cycle at >=100ms
      regardless of need; a hard floor on heartbeat/response cadence.
  N4. Heartbeat is ECHOED, not generated (increment commented out, ~1961). If the
      protocol expects the app to prove liveness with its own counter, it
      currently doesn't — confirm against the PLC spec.
  N5. Partial send not handled (only <=0 checked) — negligible for 50 bytes but
      technically unguarded.

KNOWN DEFERRED ITEMS (do when code changes are next made)
  D1. CMakeLists.txt references SampleVisionaryTMini/CPS.cpp and CPS.h which DO
      NOT EXIST (belonged to a different forked project). Remove the references.
  D2. makeCommand() byte 17 bug: bit 2 (CLPS-OK to PLC) is set from PCA_isCLPS_OK
      while bits 0/1 use VA results, and it ignores the combined bClpsOkDetected.
  D3. Code-quality debt: 333 KB single TU, heavy duplication in per-chassis
      point-cloud slicing, debug printf in hot loop. Stale in-tree copies:
      20250623_SampleVisionaryTMini.cpp, SampleVisionaryTMini.7z, z4vg0n0b.ybp~.


###############################################################################
# 5. BATCH-RESULT ANALYSIS
#    Source: K:\TUAS\South_Endfeed\3DSP Log\{FRONT,REAR}\batch_result
###############################################################################

Dataset: 44 job recordings (22 scenarios x FRONT + REAR cameras), 8,186 frames.
"Detected" = relevant target box had prob > 0 (genuine model output; preset
fallback is applied AFTER logging, so these are TRUE detection rates).

res.txt format: ';'-separated, 51 columns, header in row 0. Frames where the 2D
detector found nothing emit a SHORT 27-column row (VA columns omitted).
Key cols: Hole_prob(5), XT_prob(11), Guide_prob(16), Landed_Trigger(21),
Landout_Detected(22), CLPS_Detected(23), CLPS_OK(24), pca landout/clps/clpsOk
(37-39), CLPS_OUT(49), CLPS_OK_OUT(50). -10000 = sentinel "no value".


5.1  2D DETECTION RATE  by CHASSIS TYPE x CYCLE
-----------------------------------------------
  Chassis  Cycle    Frames   Container(hole)   Chassis(cone/guide)
  XT       Mount     2327         67%                92%
  XT       Offload    834         92%                79%
  IGH      Mount     3915         34%                98%
  IGH      Offload    780         28%               100%
  CST      Mount      250         34%               100%
  CST      Offload     80        100%                31%

5.2  2D DETECTION RATE  by CAMERA LOCATION
------------------------------------------
  FRONT  frames=3708   Container=50%   Chassis=94%
  REAR   frames=4478   Container=49%   Chassis=94%

5.3  2D DETECTION RATE  by JOB POSITION
---------------------------------------
  Big     frames=2459   Container=43%   Chassis=91%
  Center  frames=3896   Container=58%   Chassis=98%
  Small   frames=1831   Container=40%   Chassis=90%

5.4  2D DETECTION RATE  by CHASSIS x CAMERA
-------------------------------------------
  XT  FRONT/REAR    Container=73%/73%   Chassis=92%/86%
  IGH FRONT/REAR    Container=33%/33%   Chassis=97%/100%
  CST FRONT/REAR    Container=41%/57%   Chassis=72%/92%

DETECTION TAKEAWAYS
  - Chassis target detection is STRONG (~92-100%) for XT and IGH.
  - Container (hole) detection is the systemic WEAK spot. In MOUNT it's 34%
    (IGH/CST) to 67% (XT) — expected, since descending container/spreader
    occludes the corner-casting top-down until landing. Logic compensates with
    PCA + presets, so mount landout/deviation lean heavily on the 3D pipeline.
  - IGH OFFLOAD container = 28% is anomalous and BIMODAL (per job ~100% or ~0%):
    e.g. FRONT IGH Offload Center45/Small45 = 100%, but Big40/Big45/Center40 and
    most REAR IGH offloads = 0%. The hole class is unreliable on IGH from certain
    views — investigate (training-data gap or view geometry).
  - CST is the WEAKEST type overall (only 2 scenarios, small sample): chassis
    detection drops to 31% in offload and 72% on FRONT.
  - Specific bad jobs to inspect:
        REAR XT Offload Big  CHS40  -> chassis 0%
        REAR XT Offload Small CHS45 -> chassis 34%
        FRONT XT Offload Small CHS40-> container 12% / chassis 16%


5.5  LOGIC FIRING — LANDOUT (mount cycle)
-----------------------------------------
Fires (VA or PCA) in 7 of 19 mount jobs. Both sub-pipelines exercised:
  VA-driven : FRONT XT Center45 (351 fr), FRONT IGH Center45 (89), REAR XT Big20 (25)
  PCA-driven: REAR IGH Big45 (82), small PCA hits on REAR CST/IGH jobs
Remaining 12 mount jobs show zero landout — plausibly correct "Land-OK", but
CANNOT be confirmed without ground truth. FRONT XT Center45 shows VA AND PCA
agreeing (351/350) — redundancy working as intended.


5.6  LOGIC FIRING — CLPS / CLPS-OK (offload cycle)   *** BIGGEST CONCERN ***
---------------------------------------------------------------------------
  Chassis  Jobs  Frames   CLPS_OUT   CLPS-OK_OUT
  CST       2      80         0           30
  IGH      11     780         0            0
  XT       12     834         0            0

  - CLPS-OK fires in only 1 of 25 offload jobs (FRONT CST Center, 30 fr).
    Every XT and IGH offload produces ZERO CLPS-OK output.
  - CLPS_OUT (the actual alarm) is 0 everywhere. If the test set has no real
    lift events, zero alarms is CORRECT — but then CLPS-OK should assert on
    nearly every offload frame, and it doesn't.
  - Strongly suggests CLPS/CLPS-OK is gated by a condition almost never met in
    these recordings (spreader-state/landed combo, or PCA cone-base averaging in
    clps_detection_va / CLPS_Detection_PCA not reaching its count threshold).
    The CST job proves the path is reachable but conditions are too narrow / mis-tuned.


5.7  GAPS & RECOMMENDATIONS
---------------------------
  1. CLPS-OK is effectively non-functional across the offload set — investigate
     FIRST. Trace why _clps_ok_detected / _clps_ok_detected_pca stay false: the
     gating in clps_detection_va (AVG_CONE set condition) and deviation_output_pca,
     and whether _sprd_landed_trigger / offload-cycle gating is reached in replay.
  2. No ground-truth column in res.txt. "Performing as expected" can only be
     judged on internal consistency now. "Chassis type with Job.xlsx" (sibling of
     these folders) and the batch summary CSV
     (Total_Job;OK3;OK-VA,OK-PCA;... from OfflineDebugBatchProcessingThread)
     likely hold expected pass/fail — pairing those with this per-frame data would
     give true accuracy / false-positive rates.
  3. Improve container-hole detection for IGH (and mount generally) — it's the
     limiting factor for the 2D pipeline; the system currently survives on
     PCA + presets.
  4. Coverage is thin for CST (2 scenarios) and uneven (XT more offload, IGH more
     mount). More CST and balanced mount/offload recordings would make per-type
     conclusions statistically sound.

==============================================================================
 END OF ANALYSIS
==============================================================================
