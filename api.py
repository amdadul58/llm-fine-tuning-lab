from fastapi import FastAPI
from pydantic import BaseModel
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "distilgpt2"
ADAPTER_PATH = "outputs/lora-distilgpt2"


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Agriculture AI API",
    description="Browser-based Agriculture AI using LoRA fine-tuned DistilGPT2",
    version="1.0.0",
)


# ============================================================
# Request Schema
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load Base Model
# ============================================================

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

base_model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# Load LoRA Adapter
# ============================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("Agriculture AI model loaded successfully.")


# ============================================================
# Generate Answer
# ============================================================

def generate_answer(question: str) -> str:

    prompt = (
        "### Instruction:\n"
        + question.strip()
        + "\n\n"
        + "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return answer


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Agriculture AI API is running",
        "docs": "/docs",
        "model": "distilgpt2 + LoRA",
    }


# ============================================================
# Ask Agriculture AI
# ============================================================

@app.post("/ask")
def ask_ai(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        return {
            "success": False,
            "message": "Please enter a question.",
        }

    answer = generate_answer(question)

    return {
        "success": True,
        "question": question,
        "answer": answer,
    }