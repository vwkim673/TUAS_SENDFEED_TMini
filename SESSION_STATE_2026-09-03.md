# Session State — 2026-09-03 (handoff, self-contained)

> 컨텍스트 클리어용. 이번 세션 = CLPS 옆-cone 오검출 진단/수정 + XT FRONT VA preset 재도출.
> 관련: `SESSION_STATE_2026-09-01.md`, `CLAUDE.md`(도메인), v1.7 소스.

**버전**: `program_version="1.7"`. 빌드는 **사용자**(VS Release|x64, `build/SampleVisionaryTMini.vcxproj`).
이번 세션 중 사용자가 여러 번 리빌드함. exe `build/Release/TUAS_Endfeed_3DSP_v1.7.exe`.
**라이브 전제: `FORCE_ROI_MODE=0`, `ROI_FALLBACK=1`.**

---

## PART A — 이번 세션 코드 변경 (전부 `SampleVisionaryTMini.cpp`, 검증됨)

1. **chassis-type PLC 폴백 + VA 락** (`:9739`/`:9760`, globals `:848`, reset `:2098`) —
   VA 완전 블랙아웃(det_count==0) 시 `_detected_chassis_type`가 "unknown"으로 굳어 **ROI fallback이
   스킵**되던 버그 수정. 이제 PLC job type로 폴백(assume job order correct) → ROI 작동. VA가 몇 프레임
   타입 확정하면 `ChassisType_Locked`로 래치(백업). `CHASSIS_TYPE_LOCK_N=3`.
2. **byte16 bit0–2 게이트** (`:10536`) — bDetected_XT/CST/IGH를 `det_count>0`일 때만 세팅.
   블랙아웃 프레임엔 PLC로 unknown(전부 off) 유지. (bit4 bDetected_Chassis는 원래 VA박스 기반이라 무관.)
3. **IGH offload 3D 출력 = chassis 기준** (`:10386`) — IGH는 guide가 hole을 가려 hole 무의미 →
   offload에서도 chassis(`_cone_pos`) 위치 출력(mount position phase와 동일). XT는 hole→deviation 유지.
4. **offload 3D range 가드** (`:10124`) — offload cone도 mount의 `vaInRange`처럼 preset window 밖이면
   거부→ROI. window 밖 = 옆 chassis cone. ROI도 실패하면 cone lost(옆 것 채택 안 함).
   **검증**: 45ft C1-Locked에서 3D cone이 −720(옆) 대신 +850(원래)/−10000 유지.
5. **VA 2D preset region gate** (`:9796`, global `:623`, INI parse `:1601`) — `target_selections_va`
   직후(baseline·dev·CLPS·박스그리기 前) cone 박스 center_x가 `presetBox` center에서
   `presetBox.w/2 + VA_PRESET_GATE_MARGIN` 넘게 벗어나면 = 옆 cone → `_target_cone` 드롭. XT/IGH만,
   presetBox 0-size면 스킵, margin=0이면 비활성. INI `[SETTINGS] VA_PRESET_GATE_MARGIN`(기본 120).
   **검증**: 45ft C1-Locked에서 **가짜 VA CLPS-OK가 PLC로 나가던 것(6프레임) 완전 제거**(3잡 모두 0),
   착지 타겟은 유지, gate 23회 발동.

**CLPS 방향(사장님 결정)**: CLPS 관련은 **VA-2D 유지**(3D로 안 옮김). 옆-cone은 2D gate가 1차 차단.
3D region gate는 40ft+20 인접(가까운 두 cone)에만, 해당 센서/config에 한해 적용 예정(미착수).

---

## PART B — XT FRONT VA preset 재도출 (`process.ini`, 양쪽 파일 적용)

`K:\TUAS\South_Endfeed\3DSP Log\FRONT\batch_result`의 res.txt에서 착지(Landed=1, prob≥70) 타겟 콘
2D 박스를 config별로 median 산출. `[PRESET_FRONT]` 수정(양쪽 `build/INI` + `build/Release/INI`):

| Key | 작업 | 변경 (x;y;w;h) | center_x | 출처 |
|---|---|---|---|---|
| XT_4XF | 45B/40B/40C/20B(묶음) | 유지 40;170;120;140 | 100 | gate 마진 커버 |
| XT_4XR | 45C | 410;180;120;140 → **96;188;44;58** | 470→118 | FS3 360fr |
| XT_20F | 40S | 113;180;100;100 → **98;224;43;79** | 163→119 | CS2(FRONT-40S 직접) |
| XT_20R | 45S | 380;215;100;120 → **29;213;51;66** | 430→54 | CS3 70fr |

- XT_4XR/20R의 옛 2D 박스는 **엉뚱한 쪽(오른쪽)**을 가리켰음(검출 콘과 불일치) → 검출 위치로 수정.
- 40S: REAR 40B와 **같은 센서 CS2**(사장님 통찰 맞음). REAR 40B는 콘 데이터 없어 FRONT 40S 직접 사용.
- config별 개별값(참고): 20B cx107, 40B cx73, 40C cx90 (현재 XT_4XF로 묶임 — per-position 정밀화는
  코드로 finer key 필요, 보류).
- **파싱 검증 OK**: 인라인 코멘트가 파싱 안 깸(40S 잡이 새 preset 98,224,43,79 로드 확인).

---

## PART C — ✅ 45C/45S 2D↔3D 정합 (해결 — 처음 의심은 false alarm)

**결론: 2D↔3D 정합 확인. 3D range 수정 불필요.** (`Result45` 배치, 현재 코드/모델 실행)

| 잡 (센서) | 현재코드 3D cone_x | 해당 3D range | 정합 |
|---|---|---|---|
| 45C FRONT | **−743** | FRONT XT_4XR [−930,−510] | ✅ |
| 45S FRONT | **−677** | FRONT XT_20R [−1000,−500] | ✅ |
| 45S REAR | **+756** | REAR XT_4XR [670,1110] | ✅ |

- **FRONT 센서 = 음수, REAR 센서 = 양수 (미러). 각 센서 3D range가 각자 정확.** 같은 45S 잡을
  FRONT(−677)/REAR(+756)이 미러로 봄 → 확인.
- 처음 "부호 불일치"로 보였던 건 **옛 batch_result res.txt(3DSP Log)가 현재 코드와 반대 부호 규약**을
  쓴 착시(v1.4 메모 "res cols labeled opposite"). 현재 코드는 FRONT 45C를 −743으로 내고 FRONT range
  [−930,−510]에 정확히 들어감.
- 참고: 현재 모델은 FRONT 45C/45S의 2D 콘을 자주 놓쳐(prob<70) ROI가 3D 위치 커버(ConeSrc=3). 2D
  preset(cx117/54, 옛모델 도출)은 gate/fallback 기준으로 유효하나, 현재모델 2D 검출 위치 재확인은
  더 좋은 데이터 확보 시.

---

## PART D — 라이브 안전성 판정 (정직)

| 항목 | 라이브 |
|---|---|
| VA 2D preset gate (CLPS 옆-cone 차단) | ✅ 안전(순수 2D, 검증완료). 3D 축혼동 무관 |
| 2D preset 값 3건 + 파싱 | ✅ 검출 콘 일치, 로드 정상 |
| 4XF configs(20B/40B/40C/45B) | ✅ 2D+3D 정합 |
| **4XR/20R 3D range(45C/45S)** | ✅ **정합 확인**(Result45): FRONT −743/−677 in FRONT range,
  REAR +756 in REAR range. 미러 센서, 각 range 정확. 3D range 수정 불필요 |

**종합: 이번 세션 변경(코드5 + 2D preset3)은 라이브 안전.** 처음의 3D 부호 우려는 옛 res.txt 부호규약
착시로 판명(해결). 45C/45S는 FRONT에서 VA 2D를 자주 놓쳐 ROI가 3D 커버 — 이는 검출품질 이슈(별개),
range/정합 문제 아님.

---

## PART E — 검증 산출물 / 배치 셋업

- 배치: `K:\TUAS\South_Endfeed\SiteTests\CLPS_BatchTest\` — `Input`(원본 3개 XT C1-Locked 녹화의
  표준토큰 **junction**, 데이터 복사 아님), `Result`(가드 前), `Result2`(offload 3D 가드만),
  `Result3`(2D gate까지), `Result4`(preset 검증). 원본 = `K:\...\SiteTests\3DSP Log (1)\3DSP Log\
  S204 3DSP CLPS Test Log_20260727\...`. CLPS_BatchTest 폴더 지워도 원본 무관(junction).
- 배치 방법: seoho.ini `DEBUG_BATCH_JOB=1`, `DEBUG_BATCH_ROOT_DIR`=Input, `DEBUG_BATCH_SAVE_DIR`=새 폴더
  (배치는 기존 폴더 스킵). **매번 seoho.ini 백업/복원**함(`.clps*bak`). 모델 = onnx `20260728_TUAS_SEF_3DSP`
  (VA/ONNX 경로, 현재 dev 모델).
- 폴더명 파서는 표준토큰(Offload/pos/XT/CHS/SPRD/FRONT) 요구 → C1-Locked 원본명은 junction으로 표준화.

---

## PART F — 다음 할 일

1. **현재모델 FRONT 45C/45S 2D 검출품질** — 현재모델이 2D 콘 자주 놓침(→ROI). 2D preset(cx117/54)이
   현재모델 검출과 맞는지 재확인(더 좋은 녹화 시). (정합/range 문제 아님, 검출품질)
2. **REAR VA preset 재도출** — 이번엔 FRONT만 함. `[PRESET_REAR]`는 대부분 0;0;0;0(미설정) →
   REAR 센서 앱의 2D gate/fallback 안 먹음. REAR/batch_result에서 도출 필요.
3. 40ft+20 인접(가까운 두 cone) 3D region gate — 해당 센서/config만. 해당 녹화 확보 시.
4. XT_4XF finer key 분리(40B/40C/20B 개별 preset) — 코드 변경, 선택.
5. 실제 분리 녹화(landed→hoist→separation→cone-only)로 CLPS/CLPS-OK 재검증.

**작업 규칙**: 분석 우선, 코드변경 명시 요청 시. 빌드는 사용자. 안전(크레인 가이드 방향/부호) 확정 후 편집.
