#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# GPU 설정 - 모든 가용 GPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 두 개의 GPU 모두 사용

def parse_args():
    parser = argparse.ArgumentParser(description='성균관대학교 LLM 채팅')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='/home/skku/Desktop/skku_sllm/models/finetuned/skku-llm-20250426',
        help='파인튜닝된 모델 경로 (없으면 기본 모델 사용)'
    )
    
    parser.add_argument(
        '--base_model_path', 
        type=str, 
        default="/home/skku/Desktop/skku_sllm/models/base/Llama-3-Open-Ko-8B",
        help='기본 모델 경로'
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=128,
        help='최대 생성 토큰 수'
    )
    
    return parser.parse_args()

def format_prompt(instruction):
    """사용자 입력을 모델 입력 형식으로 변환"""
    return f"### 지시문:\n{instruction}\n\n### 응답:\n"

def get_optimal_device_map():
    """
    사용 가능한 GPU를 확인하고 최적의 device_map 반환
    """
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU 수: {num_gpus}")
    
    if num_gpus == 0:
        return "cpu"
    elif num_gpus == 1:
        return "auto"
    else:
        # 여러 개의 GPU에 모델을 분산 (층 기반 병렬화)
        return "balanced"

def load_model(args):
    """모델 및 토크나이저 로드"""
    print(f"모델 로드 중... 잠시만 기다려주세요...")
    
    # 최적 장치 맵 결정
    device_map = get_optimal_device_map()
    print(f"사용할 장치 맵: {device_map}")
    
    # GPU 메모리 정보 출력
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 메모리: {free_mem:.2f} GB")
    
    # 양자화 설정 (추론 속도 개선)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,  # 멀티 GPU에 밸런스 있게 분산
        trust_remote_code=True,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    
    # 파인튜닝된 모델이 있으면 로드
    if args.model_path:
        print(f"파인튜닝된 모델 로드 중: {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        model = base_model
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=32):
    """모델을 사용하여 응답 생성"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 응답 생성 (속도 최적화 설정)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,  # 낮은 온도로 설정
            top_p=0.9,
            top_k=10,         # top_k 추가
            num_beams=1,      # 빔 검색 없이 그리디 디코딩
            early_stopping=True,
        )
    
    # 응답 디코딩
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거
    response = response.replace(prompt, "").strip()
    
    # 명백한 오염 제거 (응답 정제)
    # 특수 문자나 이상한 언어가 시작되는 부분을 잘라냄
    cutoff_markers = ["###", "##", "```", "====", "----", "©", "«", "»", "&quot;", "Ja,", "Aber", "Das"]
    for marker in cutoff_markers:
        if marker in response:
            response = response[:response.find(marker)]
    
    # 시간 측정 종료
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"[응답 생성 시간: {generation_time:.2f}초]")
    
    return response

def print_welcome():
    """환영 메시지 출력"""
    print("\n" + "="*50)
    print("🏫 성균관대학교 특화 LLM 채팅을 시작합니다 🏫")
    print("="*50)
    print("- 성균관대학교 정보와 관련된 질문을 해보세요!")
    print("- 종료하려면 'exit' 또는 '종료'를 입력하세요.")
    print("="*50 + "\n")

def main():
    args = parse_args()
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model(args)
    
    # 성능 최적화를 위한 설정
    model.config.use_cache = True  # KV 캐시 활성화
    
    # 환영 메시지 출력
    print_welcome()
    
    # 채팅 루프
    while True:
        # 사용자 입력 받기
        user_input = input("👤 질문: ")
        
        # 종료 조건 확인
        if user_input.lower() in ["exit", "quit", "종료", "bye"]:
            print("\n🏫 대화를 종료합니다. 성균관대와 함께해 주셔서 감사합니다! 👋")
            break
        
        # 빈 입력 처리
        if not user_input.strip():
            continue
        
        # 모델 응답 생성
        prompt = format_prompt(user_input)
        response = generate_response(model, tokenizer, prompt, args.max_tokens)
        
        # 모델 응답 출력
        print(f"🤖 응답: {response}\n")

if __name__ == "__main__":
    main() 