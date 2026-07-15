# ROI Cone Extraction & Target Position — 작업 기록 (핸드오프)

XT 샤시의 cone/plate를 preset ROI에서 뜯고(**영역 추출**), 그 영역에서 **target position(플레이트 꼭지점 X,Y,Z)** 을 산출해 PLC로 보내는 로직. VA(2D 모델)가 없을 때의 백업 경로(`FORCE_ROI_MODE`).

- 오프라인 튜닝·검증: `SampleVisionaryTMini/tools/*.py`
- C++ 프로덕션: `SampleVisionaryTMini/ChassisRoi.h` + `SampleVisionaryTMini.cpp`(ProcessLogic)
- **C++는 이 환경에서 빌드 불가(PCL/CUDA 없음). VS `.sln`로 빌드. 헤더는 PCL stub로 clang syntax-check만 함.**

---

## 0. 좌표축 & 데이터

- PCL `p.x`=Gantry(좌우, side-rule 축) · `p.y`=Height(위+) · `p.z`=Trolley(깊이, cone이 돌출=**nearest-Z**). 뒤 chassis body는 Z 큼.
- GT `groundtruth.txt`의 `T,H,G` = `(x, y, z)` = `(col0, col1, col2)`.
- **클라우드 해결**(`roi_fullpipe_eval.py: resolve_pcd/load_cloud`):
  - OLD job: `OriginalData\XT\{FRONT,REAR}\<job>\<frame>\<frame>_base_converted.pcd`
  - NEW job: `…\BatchResult\<job>\<최근접 timestamp>\_base_converted.pcd` (배치 10프레임 샘플, CONLY 정지라 최근접=동일 장면, GT 일치 99~100%)
  - `_Depth.pcd`는 GT와 22%만 일치 → **금지**. NaN·(0,0,0) 제거.

## 1. 파이프라인 (crop → region → target)

```
preset_crop                      Gantry cg±300 · Trolley [ct−250,+450] · Height [−600,50]
 └ extract_cone (Stage1~3)       → cone+plate 영역 (body 일부 포함 가능)   [디버그 보존]
    ├ Stage1 cone_columns        cy±140 층에서 X-bin별 nearest-Z 돌출 → 후보 (protrude_tol=100, min_col_pts=60)
    ├ Stage2 side-rule           target_on_right: True→max-X, False→min-X
    └ Stage3 tight box           X=cone nose±150 · Z=front+200(peak-walk-back) · Y=cone_plate_yband
 └ ★TARGET (NEW)★
    ├ SOR flyer 제거             region의 고립 산점 제거(k=12,std×2, ~46pt/frame) — 위에 뜬
    │                            노이즈가 max-Y를 부풀려 upper-half 기준(mid-Y)을 밀어올리는 것 방지
    ├ upper-half                 region 자체 mid-Y=(min+max)/2 이상만 (handle/아래 plate 제거)
    ├ near-Z face                front+80 (뒤 body 제외)
    ├ pc_cone_detection_naive    near-Z·target-X 코너 (x,z)
    ├ sparse-edge rescue         face<250pt & 지지 edge가 target방향 +40mm 밖이면 코너를
    │                            edge 컬럼으로 이동 (x=col median, z=col dense near-Z endpoint)
    │                            — 희소 face에서 dense(30pt) X stage가 edge 못 미치는 문제 보정
    └ corner-height              코너 (x,z) ±60 국소 median Y (plate 표면 높이)
```

## 2. 현재 C++ 구현 상태

**`ChassisRoi.h`** (header-only, vcxproj 추가 불필요):
- `chassis_roi_extract_region()` — Stage1~3 (cone_columns/cone_z_extent/cone_plate_yband 포팅). Python과 **정확히 동일 결과 검증**(130208 n=1444, 144022 n=720).
- `chassis_roi_near_face(region, faceDepth=80, upperHalf=true)` — **upper-half(region mid-Y) + near-Z face**. 검출 입력 준비.
- `chassis_roi_corner_height(region, x, z, tol=60)` — 코너 국소 median Y.
- `ChassisRoiParams.coneY` — per-config cone height(아래 §3).

**`SampleVisionaryTMini.cpp` ProcessLogic (FORCE_ROI_MODE)**:
```
_pcCone = _roiRegion                       // ROI region은 FULL 유지
position_detection_pca(...)                // OLD 결과(full-region 검출) — 비교용
── XT ──
  face = chassis_roi_near_face(_pcCone,80)  // upper+near
  nc   = pc_cone_detection_naive(face)      // 같은 검출기, narrow 입력
  nc.y = chassis_roi_corner_height(face,nc.x,nc.z)
  _cone_pos = nc                            // NEW = production target
── tracker/deviation ──
```
- `pc_cone_detection_naive`(OLD 검출 로직) **변경 없음** — 입력만 narrow.
- 트래커(`ChassisRoiTracker`) baseline 보유, spreader landed 시 freeze.

## 3. cy (cone height) 출처 — 중요

**런타임 측정 아님. per-config preset.**
- Python: `PRESETS[key][2]` (`roi_fullpipe_eval.py`). 값: CS_Big_CHS40=−265, CS_Small_CHS40=−205, FS_Big_CHS20=−163, CS_Big_CHS20=−422 … (GT body-center 유래).
- C++: `roiPrm.coneY` ← `PRESET_POS`의 `F/R_XT_*_CY`(기본값 GT-center), `process.ini`의 `PRESET_*_3D`의 `XT_* = MAX;MIN;CONE_Y` 3번째 값으로 override.
  **CY 파싱은 throw-safe(`parseCY`)** — 레거시 INI는 3번째 필드가 주석 텍스트라 stoi가 던지면
  parseProcessINI의 단일 try 때문에 이후 preset 파싱이 전부 중단됨(버그였음, 2026-07 수정).
  두 INI(`build/INI`, `build/Release/INI`)의 XT_* 7키에 CONE_Y 3번째 값 명시 완료
  (F: −218/−148/−274/−274, R: −438/−323/−435; 4XR 값은 신규 Center-CHS45 GT에서 재유도).
- **CHS20은 휠이 cone보다 nearest-Z** → 런타임 nearest-Z 추정이 휠을 잡음. **preset cy 필수**(fallback 추정은 휠 잡음 검증됨).
- 한계: 프레임별 재측정 없음. 배치 데이터는 FRONT가 FS+CS 섞여 config당 ±100mm 퍼짐(라이브 per-lane은 일관).
- **주의: upper-half의 "center-Y"는 preset cy가 아니라 region 자체의 (min+max)/2**.

## 4. 검증 결과

**영역 추출 (vs GT body, 48 CONLY):** recall 91% · precision 91% · wrong-cone 1/48.

**Target position (NEW = SOR+upper+face+rescue, vs GT):**
- 중앙값 **|dT| 6 · |dH| 4 · |dG| ~9 mm**. (OLD full-region은 dT9/dH10/dG13.)
- 전축 20mm 이내 **36/47 (77%)**, 30mm **43 (91%)**, 40mm **45 (96%)**.
- **남은 이상치(|err|>40) 3개**:
  - FS_Big_CHS20 **215203** dT+101 → **ROI-miss**(정차 −400mm outlier, 타겟팅 아님)
  - FS_Big_CHS40 **163000**(dG−80), **060852**(dG−44): 코너 X-컬럼 안에 GT보다 ~40-80mm 앞
    구조(cone tip/립)가 실재 — 기하만으론 코너와 분리 불가. FS_Big_CHS20 144022 dH−57 잔존.
- 8개 config(CS_Big/Small, RS, Center, CHS45)는 대부분 <20mm. **Center는 GT body 과대라벨**(Z폭 수천mm) 주의.

## 5. 도구 (tools/)

- `roi_cone_extract.py --state CONLY --plate-depth 200` — 영역 추출 config별 recall/prec/wrong.
- `roi_make_comparison.py` — `Comparison\XT\<job>\` 에 `_OUR_cone`/`_OUR_crop`/`_GT_roi` 쌍.
- `roi_target_compare.py` — **3-way 결과지** `Comparison\XT_target\index.html`: 표(OLD/NEW dT,dH,dG) + 단면(🔴o GT/🟡+ OLD/🔵x NEW).
- `roi_target_steps.py` — `Comparison\XT_target\steps\<job>\<frame>_*.pcd`: `01_crop`,`02_coneRegion`,`03_candBand`,`04_cand_sel/_other`,`05_upperHalf`,`06_nearFace`,`06_tgt_OLD/NEW/GT` (CloudCompare용).

## 6. 열린 항목 / 다음 단계

1. **★Part A: 노이즈/handle robustness GATE★** (가장 중요, 미구현)
   - 문제: near-Z anchor는 **앞쪽 coherent 노이즈에 취약** — crop에 60점만 넣어도 target −120mm(검증됨).
   - 방향(검증됨): **density-onset + 150mm guard + height-signature**로 near-Z 후보가 진짜 cone인지 gate.
     - Y-Z 평면, Z-bin 30mm, per-bin (density, bar-height=Y-extent).
     - guard: density≥thr 첫 지점부터 150mm 내만.
     - cone 서명: **increasing→local max→decreasing** height.
   - **단, height local-max는 cone Y-Z 중심이라 GT G에서 81mm 벗어남 → target Z로 쓰면 안 됨.** height 증가-시작(rise-start)이 cone 앞면과 13mm로 일치하나, **단순 near-Z(dG9)가 여전히 더 정확**.
   - **결론: target Z는 near-Z 유지, density+height 서명은 "이게 cone 맞나" GATE로만.** (노이즈/handle 프레임에서만 발동.)
2. **FS_Big_CHS20/40 잔차 4프레임** — depth/lateral. step PCD로 원인 규명 필요.
3. **CHS20 215203** — 정차 −400mm outlier(crop 밖). CHS20 preset/bay 처리(별건).
4. **Center GT 과대라벨** — 재라벨(코드 아님).
5. **IGH/CST 확장** — target 정의 다름(IGH=guide top-left, CST=inner guide). XT 확정 후.
6. **process.ini per-lane cone-height** 채우기.
7. **VS 빌드 검증** (이 환경 불가).
8. (기존 deferred) `makeCommand()` byte17 CLPS-OK 버그.

## 7. 실험/결정 로그 (재시도 방지)

- 통계적 outlier 제거(SOR/MAD/percentile) — **역효과**: body-dense 프레임에선 target(near cone)이 소수라 잘려나감(CS_Small dG 60→121).
- Y-Z density×height / height local-max — target Z가 cone 중심(81mm off) → 나쁨.
- height rise-start — cone 앞면 13mm 일치하나 near-Z보다 못함.
- **near-Z + upper-half + near-face(80)** = 현재 NEW = clean 정확도 최선(dG9).
- (2026-07 검토) 코너 Z를 X-edge 컬럼에서 재측정하는 **무조건 edge-aware/tilt-fix — 실패**:
  ±60 컬럼은 정상 config에 +13~16 bias(denseExtreme=window 평균 탓)·±90 컬럼은 cone TIP을
  잡아 −40 bias. cone tip vs plate 코너를 기하만으로 판별 불가 → 무조건 보정 금지.
- **sparse-edge rescue(조건부)만 채택**: face<250pt + edge offset>40 + 컬럼≥12pt 일 때만
  발동 → 47프레임 중 정확히 2개(172200 dT−74→−1, 130210)만 변경, 회귀 0. within20 34→36.
- (2026-07) **3-plane(X-Z/Y-Z/X-Y 단면) 재설계 실험 — 기각** (`tools/roi_plane_target.py` 보존):
  P1 탑뷰 근접-Z 그룹→X범위/zF, P2 사이드뷰 yTop(z)→coneTop/plateY/zBack, P3 정면뷰
  plate-band edge→코너. 3회 튜닝에도 **w20 19/46 vs 현행 35** (dT 19 vs 7, dH 11 vs 4;
  CS_Big_CHS40 dT−123, CHS45 dH+213 구조적 잔존). **실패 원인**: 각 평면 추정치(30mm 양자화된
  그룹 extent, plateY, coneTop)가 다음 평면의 입력이 되어 **오차가 단계적으로 증폭(cascade)**;
  코너는 3D-국소 특징이라 실제 3D 부분집합에서 직접 찾는 현행 방식이 우월. 현행 파이프라인도
  사실상 평면분석의 합성(cone_columns=탑뷰, face=깊이컷, 검출기=Y슬라이스→X/Z슬라이스)이므로
  순수 2D 재구성의 이득 없음. **P2의 yTop-drop 기반 zBack은 region 트리밍용으로 재활용 가치.**
- (2026-07) **사용자 제안 파이프라인**(Y-floor→Z-front-edge창→SOR→plate-band→plateY±100→검출기)
  **5회 반복 검증** (`tools/roi_user_pipeline.py` 보존, v5): 최종 w20 **23-25/47 vs 현행 36**.
  - **확인된 것**: Z-front-edge[zc−50,+150] 창(≈near-face와 동등), plate-band("넓은 밴드 중
    최상단, 위는 좁음" 서명)는 **CS/RS 정면성 뷰에서 현행과 동급**(CS_Big_CHS20 1,1,5mm).
  - **구조적 한계 2개**: (a) **Y-floor는 cy prior를 대체 불가** — FS CHS20 휠 상단(−480)과
    plate 하단(−476)이 Y에서 겹치고 휠이 cone과 같은 X·더 가까운 Z → 어떤 floor로도 분리
    불가, X-그룹/Z-edge 모두 cy±140 prior 필요(v4/v5 하이브리드로 해결). (b) **plate-band가
    FS(사선) 뷰에서 취약** — 사선 전면은 넓은 밴드가 여러 높이에 생겨 "최상단 넓은 밴드"가
    낮은 밴드를 잡음(FS_Big_CHS20 dH−246, CHS45 +234 잔존). 현행(upper-half) 유지 권고.
- 영역을 preset으로 넓게 뜯고 targeting에서 좁히는 구조. body는 near-face로 제외, handle/아래는 upper-half로 제외.
- `pc_cone_detection_naive`의 `targetPt.y=mpz_minY`(슬라이스 바닥) 버그 → corner-height 국소 median으로 보정.
