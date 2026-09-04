# Session State — 2026-09-01 (handoff, self-contained)

> 컨텍스트 클리어용. 이 세션에서 한 일 + 현재 상태 + 다음 할 일. 관련: `CLAUDE.md`(도메인),
> `IGH_AppState.md`(IGH 이전 상태), `V1.6*_FIELD_UPDATE_REPORT.md`.

**버전**: 소스 `program_version = "1.7"` (사용자가 리빌드함, exe `TUAS_Endfeed_3DSP_v1.7.exe`).
빌드는 **사용자가 직접** (Claude는 코드/INI 편집만). 빌드 = VS Release|x64, `build/SampleVisionaryTMini.vcxproj`.

---

## PART A — v1.7 코드 변경 (필드/통신/종료, 전부 `SampleVisionaryTMini.cpp`)

이번 세션에 들어간 것들 (R2~R4 이력이 1.7로 합쳐짐). **라이브 전제 유지: `FORCE_ROI_MODE=0`, `ROI_FALLBACK=1`.**

1. **VA preset INI 토글** — `[SYSTEM] ENABLE_VA_PRESET` (기본 1=legacy). preset 박스가 (0,0)일 때
   prob=50 가짜검출로 나가던 리스크 차단용. 0=끄면 VA-miss는 -10000, 3D는 ROI_FALLBACK.
2. **저장경로 app_path 앵커** — `job_result_folder_name`("JOB_LOG/") + `saveDirName`("SAVE/") 상대경로
   → `app_path + "/..."`(exe 폴더). 런처(PM Manager/바로가기) CWD 무관하게 exe 옆에 저장. 메인로그는
   원래 `app_path/Log/`.
3. **PLC 소켓 안정화 + `[PLC]` 로그** — `socketClient`:
   - `SO_RCVTIMEO`(1000ms)+`SO_SNDTIMEO`+`SO_KEEPALIVE`. recv 타임아웃 시에도 heartbeat 계속 송신
     (죽은 소켓은 send 실패로 감지→재연결). timeout vs recv=0 vs error 구분.
   - **TCP 프레임 재조립**: 50바이트 프레임을 온전히 모아서 parseCommand (partial read가 bay 등
     필드를 어긋나게 읽어 **간헐적 bay0** 유발하던 것 수정). `partial=N` 카운터 [PLC] alive에 표시.
   - `[PLC] alive up=..s ... hb=.. dHb=+.. partial=.. enProc=..` 주기 로그(10s). **heartbeat는 PLC(서버)가
     증가, 우리는 echo만** → dHb>0 = 연결정상. INI `[SETTINGS] PLC_RECV_TIMEOUT_MS/PLC_SILENCE_WARN_MS/PLC_STATUS_LOG_MS`.
4. **Shutdown bit 정상화** (byte19[1], `[SYSTEM] ENABLE_RESTART_APPLICATION=1` 필요):
   - 게이트 **엣지(trigger_terminate)→상태(process OFF + save queue empty)**. `g_shutdownPending` 래치,
     큐 남으면 10초마다 `[SHUTDOWN] waiting...`, 비면 즉시 종료. bit 해제 시 취소. reset_recvVariables에서 리셋.
   - **종료 보장: `TerminateProcess`** (graceful join이 blocking getNextFrame에서 hang하던 것 우회).
   - 검증(v1.6R2 로그): 상태게이트+TerminateProcess 필요성 확인됨(62초 hang + 재시작후 로그만).
5. **Lane_Number 범위 가드** — `selectTargetSensor`가 `FRONT/CENTER/REAR_IP_ADDRESSES[Lane_Number-1]`를
   범위검사 없이 인덱싱 → **bay0(Lane_Number=0)면 [-1] = UB**. `laneIpSafe()`로 범위검사(빈 IP 반환),
   setup 스킵(재시도 churn 방지), 백업재연결도 `!current_lane_ip.empty()` 게이트. **bay0 원인 = PLC가
   bay 세팅 전에 New Job Pulse를 쏨(타이밍). 현장에서 PLC delay로 해결키로 함.** (enable-시 재선택은
   시도했다 **revert**함.)
6. **콘솔 QuickEdit 비활성화** — main 시작 시 `SetConsoleMode(... & ~ENABLE_QUICK_EDIT_MODE | ENABLE_EXTENDED_FLAGS)`.
   콘솔 창 클릭→QuickEdit→stdout 정지→logMessage printf 블록→전 스레드 hang(하트비트/shutdown 정지)
   방지. (현장 "메시지 창 클릭 시 행" 원인.)

**배포**: exe만 교체(INI 신규키 전부 옵션+기본값). 4개 앱. `ENABLE_RESTART_APPLICATION=1`은 각 인스턴스
seoho.ini `[SYSTEM]`에 넣어야 shutdown 실동작. PM Manager 자동재실행 필요.

---

## PART B — IGH merged 타겟팅 (신규 도메인 변경, 이번 세션 핵심)

**요청(site owner)**: 통짜 long IGH 가이드(길이 가변)는 양끝→center 기준 offset. XT/분리 IGH(40ft CS
either-end+offset)는 유지.

**방향 규칙** (`:9627`, 변경 없음): `target_on_right = (cam_pos=="Center" && (jobPos=="Big" || chassisLen==20))`.
`pc_cone_detection_naive`에서 **target_on_right=true → min-X, false → max-X** (코드 확인). CS+CHS20 Small은
없어서 chassisLen==20 항 무해.

**offset 2층** (변경 없음): ① inset(`XT_TWOCONE_INSET=130`) = 샤시/cone 극단→corner (xt_center_end_target).
② x_offset=26/z_offset=14 = 컨테이너 hole↔샤시 정렬(deviation_output_pca, target_on_right로 부호).

**추가한 것 (`SampleVisionaryTMini.cpp`)**:
- `igh_target_from_region`(`:9457`)에 **merged 분기**: 가이드 dense-Y밴드 X-span ≥ `IGH_MERGE_MIN_LEN`(300)이면
  `center=(xmin+xmax)/2`, `tx = center + (isBig? +IGH_MERGE_OFFSET : -IGH_MERGE_OFFSET)`(Big +30/Small −30),
  Y=guide-top, Z=near-Z. `[IGH_MERGE] span=.. centre=.. -> tx=..` 로그.
- `isBig` 인자 추가(forward decl `:8603` 기본 true, 정의 `:9457`). 라이브 호출부 `jobInfo.jobPos=="Big"` 전달:
  position_detection_pca IGH(idx3)/IGHS(idx4) `:8665/8680`, computeRoiConePos `:9576`.
- INI: `[SETTINGS] IGH_MERGE_MIN_LEN=300`, `IGH_MERGE_OFFSET=30` (양쪽 process.ini).

**검증 (v1.7 배치, 사용자 지정 샘플 `K:\TUAS\South_Endfeed\GroundTruth_Set\CS_40ft_Samples`)**:
- **라이브 경로(FORCE_ROI=0)에서 정상**: merged 샘플 `[IGH_MERGE] span=548 centre=799 -> tx=769 (isBig=0)`
  (Small이라 −30). 분리 샘플은 span<300 → merged 안 뜸(end-corner). XT는 2-cone(극단→−130).
- **⚠️ FORCE_ROI 배치에선 merged 놓침** — `chassis_roi_extract_region`의 xspan이 IGH region을 **~300으로
  클립**(VA 박스는 span 1018인데). 라이브는 VA박스(pcCone) 사용→OK, FORCE_ROI만 클립. **라이브 무관.**
  필요하면 `IGH_ROI_40` xspan을 넓혀 FORCE_ROI 배치 검증에서도 안 잘리게 보완 가능(미적용).

---

## PART C — 타겟팅 로직 step-by-step (현재 코드)

**XT 40 Center 2-cone — `xt_center_end_target`(`:9492`)**: SOR → near-Z tips(Z 3pct+150, upper-half) →
바깥 X극단(Big=max-X/Small=min-X) → `tx=극단∓130`(inset) → Z=극단±230 slab 20pct → Y=로컬 plate-corner.

**IGH 40 Center — `igh_target_from_region`(`:9457`)**: (0,0,0)제거+SOR → guide dense-Y밴드 X-span:
- **분리(<300)**: near-face(80) → `pc_cone_detection_naive(target_on_right)` 끝코너 → sparse-edge rescue →
  Y=guide-top. offset 없음.
- **merged(≥300)**: center=(xmin+xmax)/2 → `tx=center±30`(Big+/Small−) → Z=near-Z, Y=guide-top.

**CONLY 노트**: CONLY(컨테이너 없음)엔 container corner-casting 없음. 단 model `hole`은 샤시 twist-lock도
검출하므로 hole이 있어도 CONLY일 수 있음. merged는 **CONLY에서 span 안정**(468±2 vs 컨테이너 있을때 392-509).

---

## PART D — 시각화 툴 (`SampleVisionaryTMini/tools/`)

- **`target_overlay.py`** — 씬+타겟 컬러 마킹 PCD(XYZRGB, CloudCompare용). `_sphere`(구형 마커),
  `write_colored_pcd`, `_packrgb`. `read_pcd`(organized 512×424 순서보존, `roi_fullpipe_eval.R`) → 3D→pixel은
  nearest organized index. **주의: cloud 기반 Python 리플리카는 부정확했음** → C++ 배치 실출력 사용으로 전환.
- **`result_overlay.py`** — **C++ 배치 실출력**에서 타겟 마킹: 빨강=hole(CC), 노랑=offset 전, 초록=offset 후.
  merged raw=final∓offset(Small/Big 부호). 소스=`Result_live`(FORCE_ROI=0 라이브 타겟).
- **`region_overlay.py`** — ROI 뜯은 region 그리기: 회색=씬, **주황=VA박스만, 청록=ROI만, 마젠타=겹침(둘다)**,
  노랑/초록=타겟. region=`_ROI_ACQ_coneRegion`(FORCE_ROI), body=`_AB_va_body`(VA박스=라이브가 쓰는것).
  → merged에서 주황(VA-only)=ROI가 클립한 병합가이드 부분이 시각적으로 보임.

**샘플 데이터**: `K:\TUAS\South_Endfeed\GroundTruth_Set\` (`OriginalData/{XT,IGH}/{FRONT,REAR}/<job>/Image+Depth`
= raw; `<job>/batch_result 또는 BatchResult/<frame>_Image/_base_converted.pcd` = organized 월드 cloud;
`GroundTruth/{IGH,XT}/groundtruth.txt` = `job;frame;CONLY|LANDING|LANDED;T;H;G`). base_converted = 217088점
= 512×424 organized (index↔pixel). 사용자 지정 3샘플 = `CS_40ft_Samples/`.

**C++ 배치 검증 방법** (사용자가 리빌드 후): seoho.ini `DEBUG_BATCH_JOB=1`, `DEBUG_BATCH_ROOT_DIR/SAVE_DIR`,
process.ini `FORCE_ROI_MODE`(=0이면 라이브 타겟/=1이면 ROI region 저장). 입력=raw레이아웃(Image/+Depth/,
파일명 `..._Image.jpg`/`..._Depth.pcd`), 잡폴더명은 표준토큰(Mount/pos/chassis/CHS/len/SPRD/FRONT|REAR).
**배치는 저장폴더 있으면 스킵→새 SAVE_DIR 필수.** 배치 셋업은 `K:\...\GT_TargetCheck\`에 남아있음(Input/Result_*).
**라이브 INI는 배치 후 반드시 복원**(FORCE_ROI_MODE=0, DEBUG_BATCH_JOB=0) — 매번 `.livebak` 백업/복원함.

---

## PART E — 다음 할 일 / 열린 항목

1. **IGH merged**: 라이브 경로 검증됨. 확정 or (선택) FORCE_ROI 배치용 `IGH_ROI_40` xspan 확대.
2. **CST**: 미착수 (CLAUDE.md TASK2).
3. **FRONT 프레임갭 진단 로그** — 요청받았으나 IGH 작업으로 보류. FRONT ~2초 hang 200여회 = 센서/네트워크가
   FRONT에 프레임 간헐 지연(getNextFrame 블록, watchdog 사각지대). stream 스레드에 frame-gap 로그 추가 예정.
4. **S227/현장 인프라** — B3·B4 센서/네트워크(별개, 앱 아님).
5. 배포: v1.7 exe 4개 앱 + `ENABLE_RESTART_APPLICATION=1` 각 인스턴스 INI + PM Manager 재실행.

**작업 규칙**: 분석 우선, 코드변경은 명시 요청 시만. 빌드는 사용자. 안전관련(크레인 가이드 방향/부호)은 확정 후 편집.
