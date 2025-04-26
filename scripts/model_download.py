# 토크나이저만 다운로드하는 스크립트
from transformers import AutoTokenizer

# 기존 캐시를 무시하고 토크나이저 새로 다운로드
tokenizer = AutoTokenizer.from_pretrained(
    "beomi/KoLLaMA-7B", 
    force_download=True,  # 기존 캐시 무시하고 새로 다운로드
    local_files_only=False  # 로컬 파일 사용 안 함
)

# 토크나이저만 저장
tokenizer.save_pretrained("/home/skku/Desktop/skku_sllm/models/base/KoLLaMA-7B")