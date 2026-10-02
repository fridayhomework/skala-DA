# ESS 배터리 수명 조기 예측

초기 100사이클의 방전곡선과 운전 신호로 배터리 셀의 총 Cycle Life를 예측하는 회귀 프로젝트입니다. EDA에서 발견한 관계를 피처와 모델로 구현하고, 다른 Batch에서도 예측 성능이 유지되는지 평가했습니다.

## 프로젝트 개요

- 데이터: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 분석 단위: 배터리 셀 1개 = 표의 1행
- 개발: Batch 1, 46셀
- 최종 Test: Batch 2, Target 유효 39셀
- Batch 3: DAY 1 EDA 비교에만 사용; 추가 모델 평가는 생략
- Target: `cycle_life`, 학습 시 자연로그 변환 후 예측을 Cycle 단위로 복원
- 주지표: MAPE (%), 보조지표: MAE·RMSE

원본 Cycle Life 라벨은 명목용량 1.1 Ah의 80%인 0.88 Ah 기준과 연결되며, 일부 라벨은 마지막 기록 사이클+1입니다. 셀별 초기용량으로 정규화한 retention 0.8 도달 시점과 완전히 같다고 해석하지 않습니다.

## 파일 구조

```text
data/README.md                         원본 출처와 배치 방법
notebooks/01_EDA.ipynb                 Batch 1·2·3 EDA
notebooks/02_feature_engineering.ipynb 피처 설계·품질·누수 점검
notebooks/03_modeling.ipynb            학습·튜닝·평가·오류 분석
src/day1_eda.py                        데이터 로딩과 공통 계산
output/day2/model_features.csv         모델링용 파생 표
output/day2/feature_manifest.csv       피처 목록과 선정 근거
results/                              성능·후보 비교·셀별 예측·설정
requirements.txt                      실행 의존성
```

## 환경 설정과 실행

Python 3.11을 사용합니다. 저장소를 다운로드하거나 clone한 뒤 루트에서 실행합니다.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name da-day1 --display-name "Python 3.11 (DA Jupyter)"
```

VS Code의 Python·Jupyter 확장을 설치하고 해당 커널을 선택합니다. 첫 셀의 `sys.executable`이 `.venv311/bin/python`을 가리키는지 확인합니다.

- 결과만 확인: 노트북의 저장된 출력과 `results/`를 읽습니다.
- 모델 재실행: 포함된 파생 CSV로 `03_modeling.ipynb`를 위에서부터 실행합니다.
- 원본부터 재실행: [데이터 안내](data/README.md)에 따라 세 MAT 파일을 준비하고 `01 → 02 → 03` 순서로 실행합니다.

그래프 한글 폰트는 macOS의 AppleGothic 기준입니다. 다른 OS에서는 설치된 한글 폰트로 `font.family`를 변경하면 됩니다. `src/day1_eda.py`가 있어야 노트북의 import가 동작하므로 폴더 구조를 유지합니다.

## EDA와 피처 설계

전체 139셀 중 Cycle Life 유효 셀은 129개입니다. Batch별 중앙 수명은 Batch 1 858.5, Batch 2 472, Batch 3 1005.5사이클로 차이가 컸습니다. Batch 1에는 500사이클 미만 셀이 없지만 Batch 2에는 유효 39셀 중 28셀이 존재했습니다.

열화곡선은 초기의 완만한 변화와 이후 가속을 보여 Knee를 탐색했습니다. Knee는 전체 곡선에서 계산하는 사후 지표이므로 조기 예측 입력에서는 제외했습니다.

| Batch | Target 유효 셀 | 단수명 <500 | 장수명 >1,000 | Knee 중앙값 (Cycle) |
|---|---:|---:|---:|---:|
| 1 | 46 | 0 (0.0%) | 10 (21.7%) | 610.5 |
| 2 | 39 | 28 (71.8%) | 3 (7.7%) | 353.0 |
| 3 | 44 | 0 (0.0%) | 23 (52.3%) | 785.0 |

수명 대비 Knee 위치의 중앙값은 각각 약 73.4%, 77.5%, 76.5%였습니다. DAY 1의 탐색적 기울기 비교에서 Knee 이후 열화속도 중앙값은 이전의 약 8배였습니다. Knee는 평활화와 곡선 근사 방법에 의존하는 탐색 지표이며 실제 물리적 전환점의 정밀 측정값은 아닙니다.

충전정책별로도 수명이 달랐습니다. Batch 1의 `4C(80%)-4C`는 평균 1,226.5사이클 (2셀), `3.6C(80%)-3.6C`는 평균 1,182.0사이클 (3셀), `8C(15%)-3.6C`는 평균 1,008.5사이클 (2셀)이었습니다. 첫 C-rate뿐 아니라 전환 SOC와 이후 충전속도의 조합을 고려해야 하며, 정책별 표본이 작아 평균 차이를 충전속도의 인과적 효과로 단정하지 않았습니다.

Cycle 100−10의 ΔQ(V) 통계량이 가장 강한 조기 신호였습니다. 전체 Spearman 상관은 `dq_log_var` 약 −0.892, `dq_min` 약 +0.883이며 Batch별 방향도 일관됐습니다. 첫 C-rate 단독 상관은 약 −0.027로 약해 충전속도 하나로 수명을 설명할 수 없었습니다. ΔQ 피처끼리도 강하게 중복돼 단일 피처와 정규화 모델을 비교했습니다.

핵심 후보는 ΔQ log variance·최솟값, 보조 후보는 초기 QD 수준·변화율, IR 수준·변화량, 온도, 충전시간과 충전정책의 수치 조건입니다. 총 15개 후보를 저장하고 1·2·5·15개 피처 조합을 비교했습니다. Batch 2의 6셀에서 초기 IR이 0으로 기록돼 미측정값으로 판단하고 NaN으로 처리했습니다. 셀을 삭제하지 않고 Train 중앙값 대체를 적용했습니다.

`cell_id`, `batch`, 원본 `policy`는 추적·분할용입니다. Target, Knee, 전체 기록 길이와 100사이클 이후 값은 입력에서 제외했습니다.

## 모델 개발과 선택 근거

Batch 1을 충전정책 그룹 단위로 Train 33셀·Valid 13셀로 분리했습니다 (`random_state=42`). 정책은 서로 겹치지 않습니다. Train 내부에서 정책 그룹 5-Fold CV와 Grid Search로 설정을 선택했습니다. 중앙값 대체·표준화는 Pipeline 안에서 각 학습 fold에만 fit합니다.

중앙값 Baseline, Ridge, ElasticNet, Huber, Gradient Boosting을 비교했습니다. Ridge·ElasticNet은 다중공선성과 소표본, Huber는 큰 잔차의 영향, Gradient Boosting은 비선형 관계를 비교하기 위한 후보입니다. 규제 강도는 여러 크기로 탐색하고 트리는 깊이 1·2로 제한했습니다. 구체적인 탐색 범위는 03의 `model_specs`에 있습니다.

선택 규칙은 최저 Valid MAPE에서 1%p 이내인 후보 중 피처 수가 적은 조합을 우선하고, 동률이면 Valid−CV Gap과 CV MAPE를 오름차순 비교하는 것입니다. 1%p는 최종 Test 평가 전에 정한 단순성 우선 허용폭이며 통계적 유의성 기준은 아닙니다. 음의 Gap 자체가 안정성을 보장하지는 않습니다. 최종 모델은 **`dq_log_var` 단일 피처의 Gradient Boosting**입니다.

- 트리 수 100, 학습률 0.1, 최대 깊이 1, 최소 leaf 표본 2
- Valid MAPE: 단일 ΔQ 5.885%, ΔQ 2개 6.063%, 대표 보조 포함 7.325%, 전체 15개 7.847%
- 중앙값 Baseline Valid MAPE 14.203%보다 개선

이 분할에서는 보조 피처 추가가 개선으로 이어지지 않았습니다. 보조 피처가 항상 무의미하다는 결론은 아닙니다. 모델과 설정을 고정한 뒤 Batch 1 전체로 재학습하고 Batch 2를 평가했습니다. Test 결과를 근거로 모델을 다시 선택하지 않았습니다.

## 성능 결과

| 구분 | MAPE (%) | 해석 |
|---|---:|---|
| Train (Batch 1 CV) | 8.890 | Train 33셀 내부 정책 그룹 CV |
| Valid (Batch 1 Hold-out) | 5.885 | 독립 정책 13셀 |
| Test (Batch 2) | 30.437 | 확정 모델의 외부 Batch 평가 |
| Gap (Train-Valid) | −3.005 | Valid − CV, %p |
| Gap (Valid-Test) | +24.552 | Test − Valid, %p |
| Gap (Target-Test) | +21.337 | Test − 원논문 9.1%, %p |

Test MAE는 약 151.7사이클, RMSE는 약 169.7사이클입니다. 원논문 9.1% 성능은 재현하지 못했습니다. 전처리·피처·표본·분할이 원논문과 동일하지 않으므로 완전히 통제된 알고리즘 비교로 해석하지 않습니다.

## 오류 분석

Batch 2의 39셀 중 37셀을 과대예측했습니다. 실제 중앙 수명 472사이클에 비해 예측 중앙값은 약 630사이클입니다. 500사이클 미만 28셀의 MAPE는 약 34.50%였습니다. C07·C16은 실제 393·396사이클을 약 630으로 예측해 APE 약 60.23%·59.02%를 보였으며 같은 충전정책입니다.

Batch 1에 단수명 사례가 부족하고, 선택한 트리 모델이 학습하지 않은 수명 영역으로 외삽하기 어렵다는 점이 주요 원인 후보입니다. 핵심 입력 `dq_log_var`의 Batch 중앙값 차이는 Batch 1 IQR의 약 0.945배이고 Batch 2 약 28.2%가 Batch 1 입력범위를 벗어났습니다. 다만 범위 밖 셀의 MAPE 28.95%는 범위 안 31.02%보다 높지 않아 범위 이탈만으로 전체 실패를 설명할 수 없습니다. 같은 ΔQ 값과 수명의 관계가 Batch에 따라 달라졌을 가능성도 있습니다.

최종 모델은 IR을 사용하지 않으므로 IR 결측을 이번 오차의 직접 원인으로 해석하지 않습니다. 입력 분포와 잔차의 연관성은 원인 가설이며 인과관계 증명은 아닙니다.

## 평가의 한계와 개선 방향

- 보고한 CV 점수는 Grid Search에서 설정 선택에도 사용된 값이므로 선택 편향이 있을 수 있습니다. 별도 Hold-out과 Batch 2 결과를 함께 해석합니다.
- 16개 모델·피처 조합을 작은 Valid 13셀로 선택해 Valid 성능도 낙관적일 수 있습니다. 반복 그룹 분할 또는 nested CV가 후속 검증 방법입니다.
- DAY 1 과제에 따라 Batch 2를 이미 EDA로 관찰했으므로 완전히 사전 미관찰된 Test라고 주장하지 않습니다. DAY 2 튜닝에는 Batch 2 성능을 사용하지 않았습니다.
- 데이터 라벨에는 종료 시점의 차이와 일부 미완전 EOL 관측이 있어 실제 교체 시점과 동일하다고 단정하지 않습니다.
- 후속 개선은 다양한 lot와 단수명 사례 수집, 피처 구현 검증, Batch별 관계 점검, 예측 불확실성 평가입니다. Test 관찰 후 개선하는 경우 새로운 외부 평가 데이터가 필요합니다.

## ESS 도메인 해석

초기 ΔQ 신호는 셀 수명 선별과 점검 우선순위에 활용할 가능성이 있습니다. 그러나 이번 모델은 Batch 2 수명을 과대평가해 교체를 늦출 위험을 보였습니다. 예측만으로 교체를 결정하지 않고 BMS의 SOH·온도·저항·안전진단과 함께 판단해야 합니다. 실제 운전환경의 외부 검증과 불확실성 경고가 필요합니다.

## 참고문헌 및 수행 역할

- Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8
- 심지용: EDA 검토, 피처 의미·선정 논리 검토, 모델 개발 과정 이해 및 성능·오류 해석 검토