# Data

데이터 파일은 레포에 포함하지 않습니다. 아래 원본을 내려받은 뒤 `notebooks/01_preprocessing.ipynb`를 실행하면 학습용 파일이 생성됩니다.

## 원본 데이터
| 데이터 | 링크 | 사용 파일 |
|---|---|---|
| Fashion Product Images | https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset | `styles.csv`, `images/` |
| Fragrantica.com Fragrance Dataset | https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset | `fra_cleaned.csv` |

## 전처리 결과 (01_preprocessing 출력)
| 파일 | 행 수 | 설명 |
|---|---|---|
| `train.csv` | 21,402 | 패션 메타데이터 + `style_label` |
| `val.csv` | 3,058 | |
| `test.csv` | 6,115 | |
| `fra_preprocessed.csv` | 24,063 | 향수 정보 + `accord_sequence`, `fragrance_text` |

노트북은 Colab에서 실행된 그대로라 파일명이 `train (1).csv`처럼 되어 있는 곳이 있습니다. 경로를 환경에 맞게 바꿔 실행하세요.
