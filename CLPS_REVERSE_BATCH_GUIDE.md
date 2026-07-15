# CLPS/CLPS-OK 역재생 배치 실행 가이드 (신입용)

## 무엇을 하는 건가요?

Offload(하역) 작업에서 컨테이너가 샤시에서 **분리되는 순간**을 감지하는 로직(CLPS-OK)을
테스트합니다. 실제 offload 녹화는 전부 착지 상태에서 끝나서 분리 장면이 없기 때문에,
**Mount(적재) 녹화를 거꾸로 재생**하면 offload와 같은 순서(착지 → 들어올림 → 분리 →
콘만 보임)가 되는 것을 이용합니다.

`ENABLE_MOUNT_REVERSE=1`이면 **각 Mount 작업이 자동으로 2번 처리**됩니다:
1. **정방향** (원래 Mount 그대로) → 결과가 `SAVE_DIR/Mount_Result/<작업명>/`
2. **역방향** (Offload로 취급) → 결과가 `SAVE_DIR/Offload_Result/<작업명>/`

한 번 실행하면 두 폴더가 모두 채워지므로 따로 두 번 돌릴 필요 없습니다.

## 준비물

- 빌드된 실행파일: `build/Release/TUAS_Endfeed_3DSP_v1.2.exe` (빌드는 담당자에게 요청)
- 대상 데이터: **반드시 `작업폴더/Image/*.jpg` + `작업폴더/Depth/*.pcd` 구조**여야 합니다.
  - 예시(검증됨): `K:\TUAS\South_Endfeed\GroundTruth_Set\XT_20260622\`
  - `OriginalData\XT\...` 아래 폴더들은 구조가 달라서 **안 됩니다** (프레임별 폴더 구조).
- 지금은 **XT 타입 작업만** 대상입니다 (폴더명에 `_XT_` 포함된 것).

## 설정 (build/Release/INI/)

### seoho.ini — [DEBUG] 섹션
```ini
MODE_DEBUG=0
DEBUG_WITH_FILES=0
DEBUG_BATCH_JOB=1
DEBUG_SENSOR_POSITION=FRONT        ; FRONT 다 돌리고 나서 REAR로 바꿔 한 번 더
DEBUG_BATCH_ROOT_DIR=K:/TUAS/South_Endfeed/GroundTruth_Set/XT_20260622
DEBUG_BATCH_SAVE_DIR=K:/TUAS/South_Endfeed/Batch_result_REVCLPS_FRONT   ; ★ 매 실험 새 폴더!
ENABLE_MOUNT_REVERSE=1             ; ★ 이 플래그가 핵심
```
- **SAVE_DIR은 반드시 새(비어있는) 폴더**로 지정하세요. 이미 있는 작업 폴더는 건너뜁니다.
- REAR 돌릴 때는 `DEBUG_SENSOR_POSITION=REAR` + SAVE_DIR도 `..._REAR`로 바꿔서 다시 실행.

### process.ini — [SETTINGS] 섹션
```ini
FORCE_ROI_MODE=1        ; 이미 1로 되어 있는지 확인만
```

## 실행

1. exe 실행 (`build/Release/` 폴더에서).
2. 콘솔 로그에서 다음 확인:
   - `[DEBUG] ENABLE_MOUNT_REVERSE: Mount jobs will be replayed BACKWARDS as OFFLOAD`
   - 각 작업이 두 번 나옵니다: `Processing: <경로>` (정방향) 와
     `Processing: <경로>  [REVERSE pass]` + `[MOUNT-REVERSE] ...` (역방향)
   - 이 로그가 안 보이면 INI가 적용 안 된 것 → 설정 다시 확인.
3. 다 돌면 SAVE_DIR 아래 두 폴더가 생깁니다:
   - `Mount_Result/<작업명>/<작업명>_res.txt` — 정방향(원래 Mount) 결과
   - `Offload_Result/<작업명>/<작업명>_res.txt` — 역방향(Offload 취급) 결과

## 결과 검토 (res.txt)

**CLPS/CLPS-OK 검토 대상은 `Offload_Result/` 쪽입니다.** (`Mount_Result/`는 기존
정방향 결과 그대로 — 별도 지시가 있을 때만 검토.)
각 작업 폴더의 `*_res.txt`를 엑셀에서 열기 (구분자 `;`). 봐야 할 컬럼:

| 컬럼 | 의미 |
|---|---|
| `Landed_Trigger` | 영상 기반 착지 판정 (역재생이라 1로 시작 → 0으로 전환) |
| `EffLanded` | 게이트에 실제 쓰인 착지 신호 (배치에서는 위와 동일) |
| `CLPSOK_3D` | ★ 3D 분리 감지 (검토 대상) |
| `CLPS_3D` | 3D 샤시들림 감지 (역재생 정상 작업에선 0이어야 함) |
| `LandedH_3D` | 학습된 착지 중 높이 기준값 (−10000 = 미학습) |
| `CLPS_OK_OUT` | 최종 통합 출력 (VA/PCA/3D 합산) |

### 작업별 체크리스트 (이대로 표 만들어서 기록)

| # | 확인 항목 | 정상 기준 |
|---|---|---|
| 1 | `EffLanded`가 1로 시작해서 0으로 바뀌는가 | 바뀜 (안 바뀌면 "landed 없음"으로 기록만) |
| 2 | `CLPSOK_3D`가 **EffLanded=1인 동안 절대 1이 아닌가** | ★★ 1이면 **심각한 버그 — 즉시 보고** |
| 3 | EffLanded 1→0 전환 후 몇 프레임 만에 `CLPSOK_3D`=1? | 보통 2~15프레임. 훨씬 늦거나 안 켜지면 기록 |
| 4 | `CLPS_3D`가 1이 되는 구간이 있는가 | 정상 작업에선 0. 1이면 프레임 번호 기록 |
| 5 | `LandedH_3D`가 −10000이 아닌 값으로 채워지는가 | 착지 구간 지나면 채워져야 함 |
| 6 | 중간에 다시 EffLanded=1이 되면 `CLPSOK_3D`가 0으로 풀리는가 | 풀려야 함 (재착지 처리) |

### 결과 정리 양식 (작업당 한 줄)

```
작업폴더명 | 센서(F/R) | landed→0 프레임# | CLPSOK_3D=1 프레임# | 지연(프레임) | 이상사항
```

이상사항 예: "착지 중 CLPSOK 켜짐(프레임 12)", "CLPSOK 끝까지 안 켜짐", "LandedH −10000 유지" 등.
전부 없으면 "없음".

## 자주 하는 실수

- SAVE_DIR 재사용 → 전부 스킵되어 결과가 안 나옴. **항상 새 폴더.**
- FRONT 설정으로 REAR까지 봤다고 착각 → FRONT/ 와 REAR/ 데이터를 각각
  `DEBUG_SENSOR_POSITION` 바꿔서 **두 번 따로** 돌려야 함 (SAVE_DIR도 각각 다르게).
- `Mount_Result/`를 열어놓고 CLPS를 검토 → CLPS 검토는 **`Offload_Result/`** 쪽.
- 이름에 `Offload_`가 든 폴더는 역재생 안 됨(원래 offload라 정방향 그대로,
  결과는 `Offload_Result/`에 저장) — 정상 동작임.
- res.txt를 쉼표 구분으로 열어서 컬럼이 밀림 → 구분자는 세미콜론(`;`).
