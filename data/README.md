# 원본 데이터 안내

데이터: MIT–Stanford Battery Dataset, Severson et al. (2019).

- 데이터 다운로드: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 원논문: https://doi.org/10.1038/s41560-019-0356-8
- 원저자 데이터 처리 코드: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

다운로드 후 저장소 루트에 `archive/` 폴더를 만들고 아래 파일을 넣습니다. 파일명은 코드에서 그대로 사용합니다.

```text
archive/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
└── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

Batch 1은 모델 개발, Batch 2는 최종 평가, Batch 3는 DAY 1 배치 비교에 사용합니다. 현재 로더는 세 파일을 함께 읽으므로 원본에서 노트북을 재실행하려면 세 파일 모두 필요합니다. varcharge 데이터는 사용하지 않습니다.

원본은 대용량이므로 GitHub에 포함하지 않습니다. `output/day2/model_features.csv`는 초기 100사이클에서 계산한 파생 데이터이며, 이를 이용하면 원본 없이도 `03_modeling.ipynb`를 실행할 수 있습니다.
