# 성균관대학교 특화 LLM (SKKU-LLM)

성균관대학교 정보와 문화에 특화된 대규모 언어 모델입니다. 학생들에게 학교 관련 정보를 제공하고, 친근한 대화를 나눌 수 있도록 개발되었습니다.

## 프로젝트 구조

```
open-ko-llm/
├── data/
│   ├── raw/              # 원본 데이터셋 저장
│   └── processed/        # 전처리된 데이터셋
├── models/
│   ├── base/             # 기본 모델 저장 (Llama-3-Open-Ko-8B)
│   └── finetuned/        # 파인튜닝된 모델 저장
├── src/
│   ├── data_processing/  # 데이터 전처리 스크립트
│   ├── training/         # 훈련 관련 스크립트
│   ├── evaluation/       # 성능 평가 스크립트
│   └── utils/            # 유틸리티 함수
├── configs/              # 훈련 설정 파일
├── logs/                 # 훈련 로그
├── notebooks/            # 실험용 주피터 노트북
├── scripts/              # 실행 스크립트
├── results/              # 평가 결과 저장
├── requirements.txt      # 의존성 패키지 목록
└── README.md             # 프로젝트 설명
```

## 환경 설정

본 프로젝트는 CUDA 지원 환경에서 실행됩니다. RTX 3090 GPU 2개가 있는 환경에 최적화되어 있습니다.

### 의존성 설치

```bash
pip install -r requirements.txt
```

## 모델 훈련

### 데이터셋 준비

학습 데이터는 `/data/processed/skku_qa_instruction.jsonl` 형식으로 준비되어 있습니다. 이 데이터는 성균관대학교 관련 질문-답변 쌍을 포함하고 있습니다.

### 파인튜닝 실행

```bash
bash scripts/finetune.sh
```

이 스크립트는 LoRA 기법을 사용하여 기본 모델을 파인튜닝합니다. 훈련된 모델은 `/models/finetuned/` 디렉토리에 저장됩니다.

## 모델 사용

### 단일 질문 모드

```bash
python scripts/inference.py --model_path /path/to/finetuned/model --prompt "성균관대학교 도서관은 어디에 있나요?"
```

### 대화형 모드

```bash
python scripts/inference.py --model_path /path/to/finetuned/model --interactive
```

## 허깅페이스 업로드

훈련된 모델을 허깅페이스에 업로드하려면:

```bash
python scripts/upload_to_hub.py --model_path /path/to/finetuned/model --repo_name skku/skku-llm
```

## 성능 평가

모델의 정확도, 응답 품질 등은 평가 스크립트를 통해 측정할 수 있습니다:

```bash
python src/evaluation/evaluate.py --model_path /path/to/finetuned/model
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
