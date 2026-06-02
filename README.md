# 👗🌸 패션 이미지 기반 향수 추천 — CNN + CLIP

> **패션 이미지 한 장으로 어울리는 향수 Top-5를 추천하는 시스템.**
> CNN으로 옷의 스타일을 6개 클래스 확률로 분류하고, 색채·향수 연구 기반의 **스타일 × 향조 매핑 행렬**로 향조 점수를 만든 뒤,
> **CLIP**의 이미지-텍스트 유사도로 보정해 24,000여 개 향수 중 Top-5를 고릅니다.

| 항목 | 내용 |
|---|---|
| 소속 | 데이터 사이언스 학회 Parrot 14기 — DL 프로젝트 (3조) |
| 기간 | 3주 <!-- TODO: 2026.MM.DD ~ 2026.MM.DD --> |
| 팀 | 4명 — 이수아(팀장), 김태현, 정재윤, 황서현 |
| 최종 모델 | **EfficientNet-B0 + 매핑 행렬 + CLIP 보정** |
| 결과 | Recall@3 **0.728**, NDCG@5 **0.862** (Baseline CNN 대비 Recall@3 2.2배) |
| 스택 | Python, PyTorch, torchvision, timm, OpenAI CLIP, scikit-learn, pandas |

## 👥 팀 구성 및 역할

| 이름 | 역할 |
|---|---|
| **정재윤** | 이미지 분류 모델(ResNet-18, EfficientNet-B0) 구조 설계 <!-- TODO: 실제 담당 부분으로 구체화 --> |
| 김태현 | 이미지 분류 모델(ResNet-18, EfficientNet-B0) 구조 설계 |
| 이수아 (팀장) | 데이터 전처리, CLIP 추천 구조 설계 |
| 황서현 | 데이터 전처리, CLIP 추천 구조 설계 |

주제 탐색·데이터 조사(1주차)와 학습·테스트·발표 준비(3주차)는 팀 전체가 함께 진행했습니다.

---

## 1. 문제 정의

향수는 Top / Middle / Base 노트로 설명되지만, 향에 익숙하지 않은 사람은 이 정보만으로 향의 분위기를 떠올리기 어렵습니다.
반면 패션은 색감과 스타일로 분위기를 직관적으로 전달합니다. 이 점에 착안해 **사용자의 패션 스타일에서 향의 분위기를 추론해 향수를 추천**하는 시스템을 만들었습니다.

```
패션 이미지 ─▶ CNN ─▶ 스타일 확률 (6) ─▶ × 매핑 행렬 (6×5) ─▶ 향조 점수 (5) ─┐
     │                                                                  ├─▶ α·매핑 + (1−α)·CLIP ─▶ Top-5 향수
     └───────────▶ CLIP 이미지 임베딩 ◀─ 코사인 유사도 ─▶ 향수 텍스트 임베딩 ─┘
```

## 2. 데이터

| 데이터 | 출처 | 규모 | 사용 정보 |
|---|---|---|---|
| 패션 | [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) | 44,424 → 정제 후 30,575장 | 상품 이미지, `articleType`, `usage`, `baseColour`, `season`, `gender` |
| 향수 | [Kaggle Fragrantica.com Fragrance Dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) | 24,063개 | `mainaccord1~5`, Top / Middle / Base 노트 |

데이터 파일은 레포에 포함하지 않았습니다 → [`data/README.md`](data/README.md)

### 2-1. 패션 스타일 라벨링 ([01_preprocessing](notebooks/01_preprocessing.ipynb))
원본에는 스타일 라벨이 없어서 메타데이터 조합으로 **6개 스타일을 규칙 기반으로 정의**했습니다.

| 스타일 | 규칙 (우선순위 순) | 샘플 수 |
|---|---|---|
| Formal | `usage = formal` | 1,826 |
| Sporty | `usage = sports` | 3,626 |
| Street | `usage = casual` + 후디·스니커즈·캡·조거 등 | 546 |
| Casual | `usage ∈ {casual, smart casual}` | 21,572 |
| Romantic | 드레스·스커트·힐·사리 등 | 529 |
| Vintage | `usage = ethnic` 또는 코트·카디건·로퍼 등 + 브라운·베이지·카키 계열 색상 | 2,476 |

- Apparel, Footwear만 남기고 양말·지갑·벨트 등 스타일 판단과 무관한 품목 제거
- Stratified 분할: **Train 21,402 / Val 3,058 / Test 6,115** (70 : 10 : 20)
- Casual이 70%를 차지하는 **심한 클래스 불균형** → 클래스 가중 손실로 대응

### 2-2. 향수 텍스트 구성
- `accord_sequence`: 메인 향조 상위 5개 (예: `citrus, white floral, sweet, fresh, musky`)
- 노트 시퀀스: 발향 순서를 반영해 **top × 1, middle × 2, base × 3** 가중 반복 (잔향이 오래 남는 특성 반영)
- `fragrance_text` = `accord_sequence` + 노트 시퀀스 → CLIP 텍스트 임베딩 입력

## 3. 스타일 × 향조 매핑 ([02_style_accord_mapping](notebooks/02_style_accord_mapping.ipynb))

스타일과 향조를 **같은 2D 좌표계**에 놓고 거리로 연결했습니다.

| 축 | 스타일 좌표 근거 | 향조 좌표 근거 |
|---|---|---|
| X | Kobayashi Color Image Scale (1990): Cool ↔ Warm | Edwards Fragrance Wheel: Fresh·Woody ↔ Floral·Amber |
| Y | Soft ↔ Hard | Zarzo & Stanton (2009) 향수 지각 PCA: Light ↔ Heavy |

- 향조 5개는 향수 휠 각도를 극좌표로 변환 (r = 0.8): floral 315°, intense 45°, natural 225°, niche 157°, romantic 337°
- 유클리드 거리 `d` → 유사도 `exp(−d)` → 행 정규화 → **6 × 5 소프트 매핑 행렬** ([assets/style_accord_mapping.json](assets/style_accord_mapping.json))

| 스타일 | Top-1 향조 | Top-2 향조 |
|---|---|---|
| Romantic | floral (0.351) | romantic (0.267) |
| Casual | natural (0.302) | floral (0.206) |
| Sporty | natural (0.433) | niche (0.205) |
| Formal | intense (0.410) | romantic (0.184) |
| Street | niche (0.394) | natural (0.192) |
| Vintage | intense (0.416) | romantic (0.196) |

하나의 스타일을 하나의 향조에 고정하지 않고 확률로 나눠 주는 **소프트 매핑**이라서, CNN의 스타일 확률 벡터와 곱하면 혼합된 스타일도 자연스럽게 반영됩니다.

## 4. 스타일 분류 모델

| | Baseline CNN | ResNet-18 | **EfficientNet-B0** |
|---|---|---|---|
| 노트북 | [03](notebooks/03_baseline_cnn.ipynb) | [04](notebooks/04_resnet18.ipynb) | [05](notebooks/05_efficientnet_b0.ipynb) |
| 구조 | Conv-BN-ReLU-Pool × 3 → GAP → FC | ImageNet 사전학습 + 스타일·향조 멀티태스크 헤드 | ImageNet 사전학습 (timm) + Dropout(0.3) + FC |
| 라벨 | 규칙 기반 `style_label` | `usage` 재매핑 라벨* | 규칙 기반 `style_label` |
| 입력 | 224 | 224 | 240 → Crop 224 |
| 증강 | Flip, Rotation(15°) | — | Flip, RandomCrop, ColorJitter |
| 손실 | 클래스 가중 CE | 클래스 가중 CE (스타일 헤드) | CE |
| 학습 | Adam 1e-4, 5 epoch | Adam 5e-5, 4 epoch | Adam 1e-4, StepLR, 5 epoch |
| 분류 성능 | Test Acc 33.3% | Test Acc 21.1% | **Val Acc 91.6% (Macro F1 0.85), Test Acc 92.1%** |

\* ResNet-18은 다른 두 모델과 달리 `usage` 컬럼만으로 스타일을 다시 매핑한 라벨로 학습해, 테스트셋에서 Street 11장·Romantic 4장만 해당 라벨을 가집니다. 따라서 세 모델의 분류 성능은 같은 조건의 비교가 아닙니다. Test Acc는 `results/pred_hybrid_clip_resnet.csv`의 예측 스타일을 테스트셋 `style_label`과 비교해 계산했습니다.

EfficientNet-B0 클래스별 성능 (Validation)

| | Casual | Formal | Romantic | Sporty | Street | Vintage |
|---|---|---|---|---|---|---|
| F1 | 0.94 | 0.86 | 0.91 | 0.79 | 0.69 | 0.92 |

## 5. 추천 및 평가 ([06_clip_recommendation](notebooks/06_clip_recommendation.ipynb))

1. **매핑 점수**: 스타일 확률 × 매핑 행렬 → 향조 점수 → 향수별 향조 벡터(앞 순위 향조일수록 큰 가중치)와 코사인 유사도
2. **CLIP 점수**: CLIP ViT-B/32로 패션 이미지와 24,063개 `fragrance_text` 임베딩의 코사인 유사도
3. **최종 점수** = α · 매핑 + (1 − α) · CLIP, 기본 **α = 0.7** (0.9 / 0.7 / 0.5 비교 실험)

### 모델별 추천 성능 (Test)

| 모델 + 매핑 + CLIP | Recall@3 | NDCG@5 |
|---|---|---|
| Baseline CNN | 0.3266 | 0.3622 |
| ResNet-18 | 0.1275 | 0.1466 |
| **EfficientNet-B0** | **0.7276** | **0.8623** |

→ 분류 정확도가 높은 EfficientNet-B0이 추천 성능도 가장 높아 최종 모델로 채택

### 최종 추천 결과 ([results/top5_recommendations.csv](results/top5_recommendations.csv))
테스트 이미지 6,114장 각각에 대한 EfficientNet-B0의 예측 스타일과 Top-5 추천 향수입니다.

예측 스타일별로 추천된 향수들의 평균 향조를 계산해 보면, 매핑 행렬에서 의도한 1순위 향조와 모두 일치합니다.

| 예측 스타일 | 매핑 행렬 1순위 향조 | 추천 향수들의 평균 향조 1순위 |
|---|---|---|
| Casual | natural | natural (0.321) |
| Formal | intense | intense (0.413) |
| Romantic | floral | floral (0.354) |
| Sporty | natural | natural (0.443) |
| Street | niche | niche (0.355) |
| Vintage | intense | intense (0.437) |

> 위 추천 성능은 최종 발표 자료 기준입니다. 세 모델을 한꺼번에 평가한 노트북은 공유 과정에서 유실되어 레포에 포함하지 못했습니다.

### ResNet-18 + CLIP 추천 결과 ([results/pred_hybrid_clip_resnet.csv](results/pred_hybrid_clip_resnet.csv))
ResNet-18 노트북은 24,063개 전체가 아닌 **후보 향수 10개** 안에서 매핑 점수(0.45)와 CLIP 점수(0.55)를 결합해 Top-5를 고릅니다. 예측 스타일은 Formal 3,636장 · Casual 1,347장 · Sporty 872장 · Vintage 260장으로 Formal에 치우쳐 있고, 6,115장 중 5,696장(93%)의 Top-1이 `secret-obsession`입니다.

## 6. 회고 및 개선 방향

- **코드·결과물 관리**: 세 모델의 Recall@3·NDCG@5를 계산한 평가 코드를 메신저로 주고받다가 파일이 만료되어 레포에 포함하지 못했습니다. 다음 프로젝트에서는 처음부터 공유 GitHub 레포에서 코드와 결과를 버전 관리하려고 합니다.
- **ResNet-18의 낮은 성능 원인**: ResNet-18은 (1) 다른 모델과 다른 `usage` 재매핑 라벨, (2) 증강 없는 4 epoch 학습, (3) 행·열 순서가 맞지 않는 매핑 행렬곱, (4) 10개로 제한된 후보 향수와 향수 이름만 사용한 CLIP 텍스트로 평가되어 다른 모델과 같은 조건의 비교가 아니었습니다. 같은 라벨·같은 추천 함수로 다시 학습·평가하면 백본 자체의 성능을 공정하게 비교할 수 있습니다.
- **매핑 행렬 순서 버그 수정**: EfficientNet 데모 추천 함수에서 매핑 행렬의 행 순서(JSON 키 순서)와 모델 출력 클래스 순서가 달라 스타일이 엇갈려 곱해지는 문제를 발견해 수정했습니다. 예를 들어 Casual 0.974로 분류된 이미지에 Romantic의 향조(floral 우세)가 계산되던 것이, 수정 후에는 Casual에 맞는 natural 우세로 바뀝니다. 이름으로 인덱싱하는 CLIP 노트북의 추천 함수에는 이 문제가 없고, 최종 추천 결과(`top5_recommendations.csv`)도 위 표처럼 스타일별 향조가 정상 매핑과 일치해 **최종 결과에는 영향이 없었던 것**으로 확인됩니다.
- **규칙 기반 라벨의 한계**: 스타일 라벨이 `usage`, `articleType` 같은 메타데이터 규칙으로 만들어져, 모델은 "사람이 느끼는 스타일"보다 "규칙이 정한 카테고리"를 학습합니다. 소규모라도 사람이 직접 라벨링한 검증셋이 있으면 실제 추천 품질을 더 정확히 평가할 수 있습니다.
- **상품 이미지와 실사용 이미지의 차이**: 학습 데이터는 단일 상품 정면 사진이지만 실제 입력은 전신 코디 사진일 가능성이 높아, 코디 이미지 데이터로의 확장이 필요합니다.
- **클래스 불균형**: Casual 70%, Street·Romantic 각 2% 미만. EfficientNet에서 Street F1이 0.69로 가장 낮아, 클래스 가중 손실이나 오버샘플링을 함께 적용해 볼 수 있습니다.
- **추천 다양성 부족**: 6,114장 중 45%(2,758장)의 Top-1이 같은 향수(`the-poplin-shirt (zara)`)이고, Top-1에 등장한 향수는 24,063개 중 150개뿐입니다. 같은 스타일로 분류되면 매핑 점수가 거의 같아지는 구조 때문으로 보여, 이미 추천한 향수에 패널티를 주는 MMR 같은 재정렬로 다양성을 높일 수 있습니다.
- **α 선택 근거**: α를 0.9 / 0.7 / 0.5로 바꿔 결과를 비교했지만 정량 지표로 고르지는 않았습니다. 검증셋에서 Recall/NDCG를 기준으로 α를 정하면 더 설득력이 있습니다.

## 7. 실행 방법

모든 노트북은 **Google Colab (GPU)** 기준으로 작성되었습니다.

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
1. `01_preprocessing` → `train.csv`, `val.csv`, `test.csv`, `fra_preprocessed.csv` 생성
2. `02_style_accord_mapping` → 매핑 행렬 생성 (결과는 `assets/`에 포함)
3. `03` ~ `05` → 스타일 분류 모델 학습 (EfficientNet 노트북은 Kaggle API 키 필요)
4. `06_clip_recommendation` → Top-5 추천

## 8. 레포 구조

```
├── notebooks/
│   ├── 01_preprocessing.ipynb          # 데이터 정제, 스타일 라벨링, 분할
│   ├── 02_style_accord_mapping.ipynb   # 스타일 × 향조 매핑 행렬
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_resnet18.ipynb               # ResNet-18 학습 + CLIP 하이브리드 추천
│   ├── 05_efficientnet_b0.ipynb        # 최종 모델
│   └── 06_clip_recommendation.ipynb    # 매핑 + CLIP Top-5 추천
├── assets/style_accord_mapping.{json,csv}
├── results/
│   ├── pred_baseline_cnn.csv          # Baseline CNN 테스트 예측 확률
│   ├── top5_recommendations.csv       # EfficientNet-B0 기반 테스트셋 Top-5 추천 결과
│   └── pred_hybrid_clip_resnet.csv    # ResNet-18 + CLIP 하이브리드 추천 결과
├── docs/presentation.pdf               # 최종 발표 자료
├── data/README.md
└── requirements.txt
```

## References
- Kobayashi, S. (1990). *Color Image Scale*. Kodansha International.
- Zarzo, M., & Stanton, D. T. (2009). Understanding the underlying dimensions in perfumers' odor perception space as a basis for developing meaningful odor maps. *Attention, Perception, & Psychophysics*.
- Edwards, M. *Fragrances of the World* — Fragrance Wheel.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP).
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
