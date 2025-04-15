from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "beomi/KoLLaMA-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 모델 저장
model.save_pretrained("models/base/KoLLaMA-7B")
tokenizer.save_pretrained("models/base/KoLLaMA-7B")