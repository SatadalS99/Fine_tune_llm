# ================================================================
#  main.py — Phi-3 Mini Personal Finance Fine-Tuning
#  All functions live here. Call them from your notebook.
#
#  Usage from notebook:
#      import main
#      main.check_gpu()
#      model, tokenizer = main.load_base_model()
#      ...
# ================================================================

import os
import gc
import sys
import json
import torch
import unsloth
import textwrap
import subprocess

import pandas as pd
import gradio as gr
from peft import PeftModel
from trl import SFTTrainer
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments


# ================================================================
#  ⚙️  CONFIG — edit here, nothing else needs changing
# ================================================================

# Model
MODEL_NAME   = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LEN  = 2048
LOAD_IN_4BIT = True

# LoRA
LORA_RANK    = 16
LORA_ALPHA   = 16
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

# Training
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-4
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
WARMUP_STEPS  = 20
SAVE_STEPS    = 100

# Paths
CSV_PATH      = "personal_finance_qa.csv"
CLEAN_CSV     = "personal_finance_qa_v2.csv"
ADAPTER_V1    = "phi3-finance-adapter-v1"
ADAPTER_V2    = "phi3-finance-adapter-v2"
MERGED_PATH   = "phi3-finance-merged"
OUTPUT_DIR_V1 = "./checkpoints-v1"
OUTPUT_DIR_V2 = "./checkpoints-v2"
EVAL_CSV      = "eval_results.csv"

# Inference
MAX_NEW_TOKENS     = 200
TEMPERATURE        = 0.1
REPETITION_PENALTY = 1.1
SYSTEM_MSG = (
    "You are a personal finance assistant. "
    "You answer questions about budgeting, saving, investing, "
    "credit, debt, and taxes clearly and accurately. "
    "If a question is outside personal finance, politely say so."
)

# GCP
GCP_PROJECT_ID    = "project-0853cd1e-1650-41a4-bfd"
GCP_REGION        = "us-central1"
GCP_BUCKET_NAME   = "my-unique-phi3-bucket_ss"
GCP_MODEL_NAME    = "phi3-mini-finance"
GCP_ENDPOINT_NAME = "phi3-finance-endpoint"

# ── GPU options (use one pair) ──────────────────────────────
# L4  (correct name: NVIDIA_L4, machine: g2-standard-4)
# T4  (correct name: NVIDIA_TESLA_T4, machine: n1-standard-4)
# CPU (no GPU, machine: n1-standard-8, gpu_type=None, gpu_count=0)

GCP_MACHINE_TYPE  = "n1-standard-8"  # CPU fallback — always works, no quota needed
GCP_GPU_TYPE      = None             # None = CPU. "NVIDIA_L4" or "NVIDIA_TESLA_T4" for GPU
GCP_GPU_COUNT     = 0                # 0 = CPU. 1 for GPU
# NOTE: No pre-built serving image used.
# We build a custom Docker image via Cloud Build (Section 9).
# The image URI is generated dynamically in build_and_push_image().


# ================================================================
#  SECTION 1 — Environment Setup
# ================================================================

def check_gpu() -> dict:
    """
    Verify a CUDA GPU is available and print its specs.
    Raises RuntimeError if no GPU is found.

    Returns
    -------
    dict  →  {gpu_name, vram_gb, torch_version}

    Example
    -------
    info = main.check_gpu()
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ No GPU detected!\n"
            "  Colab : Runtime → Change runtime type → GPU\n"
            "  Local : install CUDA drivers."
        )

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)

    props = torch.cuda.get_device_properties(0)
    info  = {
        "gpu_name":    torch.cuda.get_device_name(0),
        "vram_gb":     round(props.total_memory / 1e9, 1),
        "torch_version": torch.__version__,
    }
    print(f"✅ GPU     : {info['gpu_name']}")
    print(f"   VRAM   : {info['vram_gb']} GB")
    print(f"   PyTorch: {info['torch_version']}")
    return info


def install_libraries() -> None:
    """
    Install all required pip packages (training + UI + GCP).
    Safe to run multiple times.

    Example
    -------
    main.install_libraries()
    """
    packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "bitsandbytes>=0.43.0", "peft>=0.10.0", "trl>=0.8.6",
        "accelerate>=0.30.0",  "datasets>=2.18.0",
        "transformers>=4.40.0", "gradio>=4.26.0",
        "google-cloud-aiplatform>=1.49.0", "google-cloud-storage>=2.16.0",
    ]
    print("📦 Installing libraries …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--upgrade", "--no-deps", "--force-reinstall",
                           "--no-cache-dir", "unsloth", "unsloth_zoo"])
    print("✅ All libraries installed!")


# ================================================================
#  SECTION 2 — Model Loading
# ================================================================

def load_base_model(
    model_name:  str  = MODEL_NAME,
    max_seq_len: int  = MAX_SEQ_LEN,
    load_in_4bit: bool = LOAD_IN_4BIT,
) -> tuple:
    """
    Download and load Phi-3 Mini in 4-bit quantization.

    Returns
    -------
    (model, tokenizer)

    Example
    -------
    model, tokenizer = main.load_base_model()
    """
    print(f"🧠 Loading {model_name} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = max_seq_len,
        dtype          = None,
        load_in_4bit   = load_in_4bit,
    )
    print(f"✅ Model loaded  (dtype={model.dtype})")
    return model, tokenizer


def apply_lora(
    model,
    rank:    int   = LORA_RANK,
    alpha:   int   = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    targets: list  = LORA_TARGETS,
):
    """
    Attach LoRA adapters — only ~1-5% of params become trainable.

    Returns
    -------
    model  (PEFT-wrapped)

    Example
    -------
    model = main.apply_lora(model)
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = rank,
        lora_alpha                 = alpha,
        lora_dropout               = dropout,
        target_modules             = targets,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA applied — {trainable:,} / {total:,} params trainable "
          f"({100*trainable/total:.2f}%)")
    return model


def load_model_with_adapter(
    adapter_path: str,
    model_name:   str = MODEL_NAME,
    max_seq_len:  int = MAX_SEQ_LEN,
) -> tuple:
    """
    Load base model and attach a saved LoRA adapter.
    Use this to resume from a checkpoint without retraining.

    Returns
    -------
    (model, tokenizer)

    Example
    -------
    model, tokenizer = main.load_model_with_adapter("phi3-finance-adapter-v1")
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"❌ Adapter not found at '{adapter_path}'.")
    model, tokenizer = load_base_model(model_name, max_seq_len)
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"✅ Adapter loaded from: {adapter_path}")
    return model, tokenizer


def load_merged_model(
    merged_path: str = MERGED_PATH,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple:
    """
    Load a fully merged model (base + adapter) ready for inference.

    Returns
    -------
    (model, tokenizer)

    Example
    -------
    model, tokenizer = main.load_merged_model()
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = merged_path,
        max_seq_length = max_seq_len,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    print(f"✅ Merged model loaded from: {merged_path}")
    return model, tokenizer


def free_model(model) -> None:
    """
    Delete model from GPU and clear CUDA cache.
    Call before reloading a fresh model for retraining.

    Example
    -------
    main.free_model(model)
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared.")


# ================================================================
#  SECTION 3 — Dataset
# ================================================================

_PHI3_TEMPLATE = (
    "<|user|>\n{question}<|end|>\n"
    "<|assistant|>\n{answer}<|end|>\n"
)


def load_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load Q&A CSV, drop nulls, and print a preview.
    Expected columns: Question, Answer.

    Returns
    -------
    pd.DataFrame

    Example
    -------
    df = main.load_csv()
    print(df.head())
    """
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} rows  |  Columns: {list(df.columns)}")
    nulls = df.isnull().sum().sum()
    if nulls:
        print(f"   ⚠️  {nulls} empty cells dropped.")
        df = df.dropna()
    print(f"✅ {len(df)} clean rows ready.")
    return df


def add_fixes(df: pd.DataFrame, fixes: list) -> pd.DataFrame:
    """
    Append corrected Q&A pairs to the dataset.

    Parameters
    ----------
    df    : existing DataFrame
    fixes : list of (question, answer) tuples

    Returns
    -------
    pd.DataFrame

    Example
    -------
    fixes = [("Why are payday loans dangerous?", "They charge 300%+ APR …")]
    df    = main.add_fixes(df, fixes)
    """
    if not fixes:
        print("ℹ️  No fixes — using original data.")
        return df
    fix_df = pd.DataFrame(fixes, columns=["Question", "Answer"])
    df     = pd.concat([df, fix_df], ignore_index=True)
    print(f"✅ {len(fixes)} fix(es) added. Total: {len(df)} rows.")
    return df


def save_csv(df: pd.DataFrame, path: str = CLEAN_CSV) -> None:
    """
    Save the DataFrame to CSV.

    Example
    -------
    main.save_csv(df)
    """
    df.to_csv(path, index=False)
    print(f"✅ Saved '{path}' ({len(df)} rows).")


def build_hf_dataset(
    df:        pd.DataFrame,
    test_size: float = 0.1,
    seed:      int   = 42,
) -> tuple:
    """
    Format DataFrame into Phi-3 chat template and split train/eval.

    Returns
    -------
    (train_data, eval_data)

    Example
    -------
    train_data, eval_data = main.build_hf_dataset(df)
    """
    def _fmt(row):
        return {"text": _PHI3_TEMPLATE.format(
            question=str(row["Question"]).strip(),
            answer  =str(row["Answer"]).strip(),
        )}
    hf    = Dataset.from_pandas(df).map(_fmt)
    split = hf.train_test_split(test_size=test_size, seed=seed)
    train_data, eval_data = split["train"], split["test"]
    print(f"✅ Dataset  —  train: {len(train_data)}  |  eval: {len(eval_data)}")
    print(f"\n── Sample prompt ──\n{train_data[0]['text']}")
    return train_data, eval_data


# ================================================================
#  SECTION 4 — Training
# ================================================================

def build_trainer(
    model, tokenizer, train_data, eval_data,
    output_dir:    str   = OUTPUT_DIR_V1,
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size:    int   = BATCH_SIZE,
    grad_accum:    int   = GRAD_ACCUM,
) -> SFTTrainer:
    """
    Configure and return an SFTTrainer. Does NOT start training.

    Example
    -------
    trainer = main.build_trainer(model, tokenizer, train_data, eval_data)
    """
    trainer = SFTTrainer(
        model              = model,
        processing_class   = tokenizer,
        train_dataset      = train_data,
        eval_dataset       = eval_data,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        dataset_num_proc   = 2,
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_steps                = WARMUP_STEPS,
            num_train_epochs            = num_epochs,
            learning_rate               = learning_rate,
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            logging_steps               = 10,
            optim                       = "adamw_8bit",
            weight_decay                = 0.01,
            lr_scheduler_type           = "linear",
            seed                        = 42,
            output_dir                  = output_dir,
            # ── Checkpoint saving disabled ─────────────────────
            # Unsloth patches PyTorch internals which breaks the
            # pickler when saving mid-training checkpoints.
            # Fix: save_strategy="no" skips all mid-training
            # saves. Use main.save_adapter() after training —
            # it uses Unsloth's own safe save method instead.
            save_strategy               = "no",
            save_steps                  = 99999,
            save_total_limit            = 0,
            # ──────────────────────────────────────────────────
            eval_strategy               = "epoch",
            report_to                   = "none",
        ),
    )
    print(f"✅ Trainer ready  —  effective batch: {batch_size * grad_accum}  "
          f"|  epochs: {num_epochs}  |  lr: {learning_rate}")
    return trainer


def run_training(trainer: SFTTrainer) -> dict:
    """
    Launch training and return a metrics dict.

    Returns
    -------
    {"final_loss", "runtime_min", "samples_per_s", "peak_vram_gb"}

    Loss guide:
        Start ~2.5-3.5  →  End ~0.3-0.8  ✅
        End < 0.1       →  possible overfit ⚠️

    Example
    -------
    metrics = main.run_training(trainer)
    print(metrics)
    """
    print("🚀 Training started …  (~30 min A100 / ~60 min T4)\n")

    # ── Block mid-training checkpoints ───────────────────────
    # Unsloth overrides _save_checkpoint and calls it despite
    # save_strategy="no". The pickler then fails on SFTConfig.
    # Monkeypatching _save_checkpoint to a no-op is the only
    # reliable fix — the adapter is saved manually afterwards
    # via main.save_adapter() which uses Unsloth's safe method.
    trainer._save_checkpoint = lambda model, trial, **kw: None
    # ─────────────────────────────────────────────────────────

    stats   = trainer.train()
    vram_gb = torch.cuda.max_memory_reserved() / 1e9
    metrics = {
        "final_loss":    round(stats.metrics["train_loss"], 4),
        "runtime_min":   round(stats.metrics["train_runtime"] / 60, 1),
        "samples_per_s": round(stats.metrics["train_samples_per_second"], 2),
        "peak_vram_gb":  round(vram_gb, 2),
    }
    print(f"\n── Training complete ──")
    for k, v in metrics.items():
        print(f"   {k:<18}: {v}")
    return metrics


def save_adapter(model, tokenizer, path: str) -> None:
    """
    Save LoRA adapter weights + tokenizer (~50 MB).

    Example
    -------
    main.save_adapter(model, tokenizer, "phi3-finance-adapter-v1")
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"✅ Adapter saved → {path}/")
    for f in sorted(os.listdir(path)):
        print(f"   {f:<42}  {os.path.getsize(f'{path}/{f}') / 1e6:.1f} MB")


# ================================================================
#  SECTION 5 — Inference
# ================================================================

def ask(
    model, tokenizer, question: str,
    max_new_tokens:    int   = MAX_NEW_TOKENS,
    temperature:       float = TEMPERATURE,
    repetition_penalty:float = REPETITION_PENALTY,
) -> str:
    """
    Ask the model a single question and return the answer.

    Example
    -------
    answer = main.ask(model, tokenizer, "What is the 50/30/20 rule?")
    print(answer)
    """
    FastLanguageModel.for_inference(model)
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            do_sample          = True,
            repetition_penalty = repetition_penalty,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_turn(
    model, tokenizer,
    user_message: str,
    history:      list,
    system_msg:   str = SYSTEM_MSG,
    max_new_tokens: int = 300,
    temperature:  float = 0.2,
) -> tuple:
    """
    One turn of a multi-turn conversation.

    Parameters
    ----------
    history : list of [user, assistant] pairs (pass [] to start fresh)

    Returns
    -------
    (reply, updated_history)

    Example
    -------
    reply, history = main.chat_turn(model, tokenizer, "How do I budget?", [])
    reply, history = main.chat_turn(model, tokenizer, "Give an example.", history)
    """
    FastLanguageModel.for_inference(model)
    prompt = f"<|system|>\n{system_msg}<|end|>\n"
    for u, a in history:
        prompt += f"<|user|>\n{u}<|end|>\n<|assistant|>\n{a}<|end|>\n"
    prompt += f"<|user|>\n{user_message}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            do_sample          = True,
            repetition_penalty = REPETITION_PENALTY,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply      = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append([user_message, reply])
    return reply, history


def run_sanity_test(model, tokenizer) -> None:
    """
    Ask 3 quick questions to verify the model is working after training.

    Example
    -------
    main.run_sanity_test(model, tokenizer)
    """
    questions = [
        "What is the 50/30/20 rule?",
        "How does compound interest work?",
        "What is the best way to save money?",
    ]
    print("🧪 Sanity test …\n")
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(model, tokenizer, q)}")
        print("-" * 60)
    print("✅ Sanity test done.")


# ================================================================
#  SECTION 6 — Evaluation (30 prompts)
# ================================================================

EVAL_PROMPTS = [
    # Budgeting
    "Explain the 50/30/20 budgeting rule in simple terms.",
    "What is zero-based budgeting and how does it work?",
    "What is a sinking fund and when should I use one?",
    "What is lifestyle inflation and why is it a problem?",
    "How do I make a budget if my income changes every month?",
    # Compound Interest
    "What is compound interest and why does it matter?",
    "Explain the Rule of 72 with an example.",
    "What is the difference between APR and APY?",
    "Why is starting early so important when investing $200/month at 8%?",
    "Why is credit card debt so dangerous compared to other debt?",
    # Credit Scores
    "What is a FICO score and what range is considered good?",
    "What are the five factors that make up a credit score?",
    "What is credit utilization and what percentage should I aim for?",
    "What is the difference between a hard inquiry and a soft inquiry?",
    "How can someone build credit with no credit history?",
    # Emergency Funds
    "How much should I keep in an emergency fund?",
    "Where is the best place to keep an emergency fund?",
    "What counts as a real financial emergency?",
    "How do I rebuild my emergency fund after using it?",
    "Should I invest my emergency fund in stocks for better returns?",
    # Index Funds
    "What is an index fund and why do experts recommend them?",
    "What is the difference between an ETF and a mutual fund?",
    "What is dollar-cost averaging and how does it reduce risk?",
    "Why do most active fund managers fail to beat the market?",
    "What is the historical average annual return of the S&P 500?",
    # Debt Management
    "What is the difference between the debt snowball and debt avalanche?",
    "What is a debt-to-income ratio and why do lenders care about it?",
    "Why are payday loans considered predatory?",
    "What does it mean to be underwater on a car loan?",
    "Is all debt bad? Give examples of good and bad debt.",
]


def run_evaluation(model, tokenizer) -> list:
    """
    Run all 30 evaluation prompts through the model.

    Returns
    -------
    list of dicts: [{id, question, response, score=None}, …]

    Example
    -------
    results = main.run_evaluation(model, tokenizer)
    """
    results = []
    print(f"🧪 Running {len(EVAL_PROMPTS)}-prompt evaluation …\n")
    for i, q in enumerate(EVAL_PROMPTS, 1):
        print(f"  [{i:02d}/30] {q[:65]} …")
        results.append({
            "id": i, "question": q,
            "response": ask(model, tokenizer, q, max_new_tokens=200),
            "score": None,
        })
    print("\n✅ All responses collected.")
    return results


def print_all_responses(results: list) -> None:
    """
    Pretty-print all responses for manual scoring.
    Score guide: 2=correct  1=partial  0=wrong/refused

    Example
    -------
    main.print_all_responses(results)
    """
    for r in results:
        print(f"\n{'='*70}")
        print(f"Q{r['id']:02d}: {r['question']}")
        print(f"{'-'*70}")
        print(textwrap.fill(r["response"], 88))
        print("Score (0 / 1 / 2): ___")


def apply_scores(results: list, scores: list) -> dict:
    """
    Attach manual scores to results and print a summary.

    Parameters
    ----------
    scores : list of 30 ints (0, 1, or 2)

    Returns
    -------
    {"average_score", "pass_rate_pct", "passed", "failed"}

    Target: pass_rate_pct >= 75 before retraining.

    Example
    -------
    scores  = [2, 2, 1, 0, 2, ...]   # 30 values
    summary = main.apply_scores(results, scores)
    """
    if len(scores) != 30:
        raise ValueError(f"Need 30 scores, got {len(scores)}.")
    for r, s in zip(results, scores):
        r["score"] = s
    entered = [s for s in scores if s is not None]
    if len(entered) < 30:
        print(f"ℹ️  Only {len(entered)}/30 filled.")
        return {}
    avg    = sum(entered) / 30
    pct    = avg / 2 * 100
    passed = sum(1 for s in entered if s >= 1)
    failed = sum(1 for s in entered if s == 0)
    print(f"\n{'='*50}\n EVALUATION RESULTS\n{'='*50}")
    print(f" Average : {avg:.2f}/2.00  |  Pass rate: {pct:.1f}%")
    print(f" ≥1 (ok) : {passed}/30  |  0 (fail): {failed}/30")
    print(" " + ("✅ Above 75% — proceed to merge." if pct >= 75
                 else "⚠️  50-75% — fix failures & retrain." if pct >= 50
                 else "❌ Below 50% — review dataset quality."))
    pd.DataFrame(results).to_csv(EVAL_CSV, index=False)
    print(f" 📄 Saved → {EVAL_CSV}")
    return {"average_score": round(avg,2), "pass_rate_pct": round(pct,1),
            "passed": passed, "failed": failed}


def list_failures(results: list) -> list:
    """
    Print and return all questions that scored 0 (your fix targets).

    Example
    -------
    failures = main.list_failures(results)
    """
    failures = [r for r in results if r.get("score") == 0]
    print(f"\n❌ {len(failures)} question(s) scored 0:\n{'='*60}")
    for r in failures:
        print(f"\nQ{r['id']:02d}: {r['question']}")
        print(textwrap.fill(r["response"][:300], 80))
        print("-" * 60)
    return failures


# ================================================================
#  SECTION 7 — Merge LoRA into Base Model
# ================================================================

def merge_adapter(
    model, tokenizer,
    merged_path: str = MERGED_PATH,
    save_method: str = "merged_16bit",
) -> str:
    """
    Merge LoRA weights into the base model → single standalone file.
    T4 users: if OOM, try save_method="merged_4bit".

    Returns
    -------
    merged_path (str)

    Example
    -------
    path = main.merge_adapter(model, tokenizer)
    """
    print(f"🔗 Merging → {merged_path}  ({save_method}) …")
    model.save_pretrained_merged(merged_path, tokenizer, save_method=save_method)
    size = sum(os.path.getsize(os.path.join(merged_path, f))
               for f in os.listdir(merged_path)) / 1e9
    print(f"✅ Merged model saved  ({size:.1f} GB)")
    return merged_path


def push_to_hub(
    model, tokenizer,
    hf_username: str,
    repo_name:   str,
    hf_token:    str,
    save_method: str = "merged_16bit",
) -> str:
    """
    Push merged model to Hugging Face Hub.
    Token: https://huggingface.co/settings/tokens

    Returns
    -------
    Hub URL (str)

    Example
    -------
    url = main.push_to_hub(model, tokenizer, "alice", "phi3-finance", "hf_xxx")
    """
    repo = f"{hf_username}/{repo_name}"
    model.push_to_hub_merged(repo, tokenizer, save_method=save_method, token=hf_token)
    url  = f"https://huggingface.co/{repo}"
    print(f"✅ Uploaded → {url}")
    return url


# ================================================================
#  SECTION 8 — Gradio Chat UI
# ================================================================

def launch_ui(model, tokenizer, share: bool = True, port: int = 7860) -> None:
    """
    Launch a Gradio chat UI.
    share=True → free public HTTPS link (72 hrs).

    Example
    -------
    main.launch_ui(model, tokenizer)
    """
    def _chat(user_message, history):
        _, history = chat_turn(model, tokenizer, user_message, history)
        return "", history

    examples = [
        ["What is the 50/30/20 rule?"],
        ["How does compound interest work?"],
        ["What is a good credit score?"],
        ["Should I pay off debt or invest first?"],
        ["What is an index fund and why is it popular?"],
        ["How big should my emergency fund be?"],
    ]

    with gr.Blocks(title="💰 Personal Finance Assistant") as demo:
        gr.Markdown(
            "# 💰 Personal Finance Assistant\n"
            "Powered by **Phi-3 Mini** fine-tuned on 400+ Q&A pairs."
        )
        chatbot = gr.Chatbot(height=480)
        with gr.Row():
            txt = gr.Textbox(placeholder="Ask a finance question …",
                             show_label=False, scale=9)
            btn = gr.Button("Send", variant="primary", scale=1)
        gr.Examples(examples=examples, inputs=txt, label="Try:")
        clear = gr.Button("🗑 Clear", size="sm")
        btn.click(_chat,  [txt, chatbot], [txt, chatbot])
        txt.submit(_chat, [txt, chatbot], [txt, chatbot])
        clear.click(lambda: ([], []), None, [chatbot, chatbot])

    print("🚀 Launching Gradio UI …")
    demo.launch(share=share, server_port=port, debug=False)


# ================================================================
# ================================================================
#  SECTION 9 — GCP Vertex AI Deployment
#
#  Strategy: Custom Container (most reliable for local HF models)
#  ─────────────────────────────────────────────────────────────
#  Why NOT pre-built TGI/DLC images:
#    • TGI image tags frequently break or get removed
#    • pytorch-inference DLC requires .mar (TorchServe) format
#    • Both cause FailedPrecondition errors with HF safetensors
#
#  What we do instead:
#    1. Write a tiny FastAPI server  (serve.py)
#    2. Write a Dockerfile
#    3. Build & push image to Artifact Registry
#    4. Register model with that custom image (no artifact_uri)
#    5. Deploy → predict → teardown
#
#  Prerequisites (run once in terminal):
#    gcloud auth login
#    gcloud auth application-default login
#    gcloud config set project YOUR_PROJECT_ID
#    gcloud services enable aiplatform.googleapis.com \
#        artifactregistry.googleapis.com cloudbuild.googleapis.com \
#        storage.googleapis.com
#    gsutil mb -l us-central1 gs://YOUR_BUCKET
# ================================================================


def _gcp():
    """Lazy-import GCP SDK — keeps startup fast when not deploying."""
    try:
        from google.cloud import storage, aiplatform
        return storage, aiplatform
    except ImportError:
        raise ImportError(
            "GCP SDK missing.\n"
            "Run: pip install google-cloud-aiplatform google-cloud-storage"
        )


# ────────────────────────────────────────────────────────────
#  STEP 1 — Upload merged model weights to GCS
# ────────────────────────────────────────────────────────────

def upload_model_to_gcs(
    local_model_dir: str = MERGED_PATH,
    bucket_name:     str = GCP_BUCKET_NAME,
    gcs_prefix:      str = "models/phi3-finance",
    project_id:      str = GCP_PROJECT_ID,
) -> str:
    """
    Upload every file in local_model_dir to GCS.
    Skips hidden cache files (e.g. .cache/huggingface/).

    Returns
    -------
    gs:// URI of the model directory (str)

    Example
    -------
    gcs_uri = main.upload_model_to_gcs()
    """
    storage, _ = _gcp()
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    print(f"📤 Uploading '{local_model_dir}' → gs://{bucket_name}/{gcs_prefix} …")
    uploaded = 0
    for root, _, files in os.walk(local_model_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path   = os.path.relpath(local_path, local_model_dir)

            # Skip HuggingFace download cache — not needed for serving
            if ".cache" in rel_path:
                continue

            blob_name = f"{gcs_prefix}/{rel_path}"
            bucket.blob(blob_name).upload_from_filename(local_path)
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"   ✓ {rel_path:<50}  {size_mb:.1f} MB")
            uploaded += 1

    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    print(f"✅ {uploaded} file(s) uploaded → {gcs_uri}")
    return gcs_uri


# ────────────────────────────────────────────────────────────
#  STEP 2 — Write serving artifacts (serve.py + Dockerfile)
# ────────────────────────────────────────────────────────────

def create_serving_artifacts(
    output_dir: str  = "./serving",
    gcs_uri:    str  = "",
    cpu_mode:   bool = True,
) -> str:
    """
    Write serve.py and Dockerfile to output_dir.
    serve.py downloads model weights from GCS at container startup
    using google-cloud-storage. More reliable than /gcs/ mount.

    cpu_mode=True  -> python:3.10-slim, float32  (default, no quota)
    cpu_mode=False -> pytorch CUDA base, float16+4bit  (GPU)

    Example
    -------
    main.create_serving_artifacts("./serving", gcs_uri, cpu_mode=True)
    """
    os.makedirs(output_dir, exist_ok=True)

    gcs_path = gcs_uri.replace("gs://", "") if gcs_uri else f"{GCP_BUCKET_NAME}/models/phi3-finance"
    bucket   = gcs_path.split("/")[0]
    prefix   = "/".join(gcs_path.split("/")[1:])

    model_load_cpu = (
        "    model = AutoModelForCausalLM.from_pretrained(\n"
        "        local_path,\n"
        "        torch_dtype = torch.float32,\n"
        "    )\n"
    )
    model_load_gpu = (
        "    model = AutoModelForCausalLM.from_pretrained(\n"
        "        local_path,\n"
        "        torch_dtype  = torch.float16,\n"
        "        device_map   = \"auto\",\n"
        "        load_in_4bit = True,\n"
        "    )\n"
    )
    model_load_block = model_load_cpu if cpu_mode else model_load_gpu

    serve_py = (
        "#!/usr/bin/env python3\n"
        "import os, logging, torch\n"
        "from pathlib import Path\n"
        "from fastapi import FastAPI, Request\n"
        "from fastapi.responses import JSONResponse\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from google.cloud import storage as gcs\n\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "log = logging.getLogger(__name__)\n\n"
        "app   = FastAPI()\n"
        "model = None\n"
        "tok   = None\n\n"
        f"GCS_BUCKET = os.environ.get('GCS_BUCKET', '{bucket}')\n"
        f"GCS_PREFIX = os.environ.get('GCS_PREFIX', '{prefix}')\n"
        "LOCAL_PATH = \"/tmp/model\"\n\n"
        "def download_model():\n"
        "    log.info(f\"Downloading from gs://{GCS_BUCKET}/{GCS_PREFIX} ...\")\n"
        "    client = gcs.Client()\n"
        "    blobs  = list(client.list_blobs(GCS_BUCKET, prefix=GCS_PREFIX))\n"
        "    if not blobs:\n"
        "        raise RuntimeError(f\"No files at gs://{GCS_BUCKET}/{GCS_PREFIX}\")\n"
        "    Path(LOCAL_PATH).mkdir(parents=True, exist_ok=True)\n"
        "    for blob in blobs:\n"
        "        rel  = blob.name[len(GCS_PREFIX):].lstrip(\"/\")\n"
        "        dest = os.path.join(LOCAL_PATH, rel)\n"
        "        os.makedirs(os.path.dirname(dest), exist_ok=True)\n"
        "        log.info(f\"  {blob.name} ({blob.size/1e6:.1f} MB)\")\n"
        "        blob.download_to_filename(dest)\n"
        "    log.info(\"Download complete.\")\n\n"
        "@app.on_event(\"startup\")\n"
        "async def load():\n"
        "    global model, tok\n"
        "    local_path = LOCAL_PATH\n"
        "    if not os.path.exists(os.path.join(local_path, \"config.json\")):\n"
        "        download_model()\n"
        "    log.info(f\"Loading model from {local_path}\")\n"
        "    tok = AutoTokenizer.from_pretrained(local_path)\n"
        + model_load_block
        + "    model.eval()\n"
        "    log.info(\"Model ready\")\n\n"
        "@app.get(\"/health\")\n"
        "def health():\n"
        "    return {\"status\": \"healthy\"}\n\n"
        "@app.post(\"/predict\")\n"
        "async def predict(request: Request):\n"
        "    body      = await request.json()\n"
        "    instances = body.get(\"instances\", [])\n"
        "    preds     = []\n"
        "    for inst in instances:\n"
        "        q      = inst.get(\"inputs\", \"\")\n"
        "        params = inst.get(\"parameters\", {})\n"
        "        mx     = int(params.get(\"max_new_tokens\", 150))\n"
        "        temp   = float(params.get(\"temperature\", 0.1))\n"
        "        rep    = float(params.get(\"repetition_penalty\", 1.1))\n"
        "        prompt = f\"<|user|>\\\\n{q}<|end|>\\\\n<|assistant|>\\\\n\"\n"
        "        enc    = tok(prompt, return_tensors=\"pt\")\n"
        "        if torch.cuda.is_available():\n"
        "            enc = {k: v.to(\"cuda\") for k, v in enc.items()}\n"
        "        with torch.no_grad():\n"
        "            out = model.generate(**enc, max_new_tokens=mx,\n"
        "                temperature=temp, do_sample=True, repetition_penalty=rep)\n"
        "        new_ids = out[0][enc[\"input_ids\"].shape[1]:]\n"
        "        answer  = tok.decode(new_ids, skip_special_tokens=True).strip()\n"
        "        preds.append({\"generated_text\": answer})\n"
        "    return JSONResponse({\"predictions\": preds})\n"
    )

    base  = "python:3.10-slim" if cpu_mode else "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
    extra = "" if cpu_mode else "    bitsandbytes>=0.43.0 \\\n"
    dockerfile = (
        f"FROM {base}\n"
        "WORKDIR /app\n"
        "RUN pip install --no-cache-dir \\\n"
        "    fastapi uvicorn[standard] \\\n"
        "    transformers>=4.40.0 \\\n"
        "    accelerate>=0.30.0 \\\n"
        "    sentencepiece \\\n"
        "    google-cloud-storage \\\n"
        + extra
        + "    torch\n"
        "COPY serve.py /app/serve.py\n"
        f"ENV GCS_BUCKET={bucket}\n"
        f"ENV GCS_PREFIX={prefix}\n"
        "ENV PORT=8080\n"
        "EXPOSE 8080\n"
        "CMD [\"uvicorn\", \"serve:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\", \"--timeout-keep-alive\", \"600\"]\n"
    )

    with open(f"{output_dir}/serve.py",   "w") as f: f.write(serve_py)
    with open(f"{output_dir}/Dockerfile", "w") as f: f.write(dockerfile)

    mode = "CPU (python:3.10-slim, float32)" if cpu_mode else "GPU (pytorch CUDA, float16+4bit)"
    print(f"Serving artifacts written to '{output_dir}/'")
    print(f"   Mode       : {mode}")
    print(f"   GCS bucket : {bucket}")
    print(f"   GCS prefix : {prefix}")
    print(f"   Model load : downloads from GCS at startup (~3-5 min)")
    return output_dir




# ────────────────────────────────────────────────────────────
#  STEP 3 — Build & push Docker image to Artifact Registry
# ────────────────────────────────────────────────────────────

def _run_streamed(cmd: list, label: str = "") -> int:
    """
    Run a subprocess and stream every line of output live.
    Returns the exit code. Never hides output.
    """
    import sys
    if label:
        print(label)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode


def build_and_push_image(
    serving_dir: str = "./serving",
    project_id:  str = GCP_PROJECT_ID,
    region:      str = GCP_REGION,
    repo_name:   str = "phi3-finance-repo",
    image_name:  str = "phi3-finance-server",
    tag:         str = "latest",
) -> str:
    """
    Build the Docker image using Cloud Build and push to Artifact Registry.
    All build output streams live — nothing is hidden.

    Returns
    -------
    Full image URI  e.g. us-central1-docker.pkg.dev/proj/repo/img:tag

    Example
    -------
    image_uri = main.build_and_push_image()
    """
    image_uri = (
        f"{region}-docker.pkg.dev/{project_id}/{repo_name}/{image_name}:{tag}"
    )

    # ── PRE-FLIGHT CHECKS ────────────────────────────────────
    print("🔍 Pre-flight checks …")

    # 1. gcloud authenticated?
    r = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE",
         "--format=value(account)"],
        capture_output=True, text=True,
    )
    account = r.stdout.strip()
    if not account:
        raise RuntimeError(
            "❌ Not authenticated.\n"
            "Run in a cell:  !gcloud auth application-default login"
        )
    print(f"   ✓ Account  : {account}")

    # 2. Correct project set?
    r = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True, text=True,
    )
    active = r.stdout.strip()
    print(f"   ✓ Project  : {active}")
    if active != project_id:
        print(f"   ↳ Switching to {project_id} …")
        subprocess.run(
            ["gcloud", "config", "set", "project", project_id], check=True
        )

    # 3. Dockerfile present?
    dockerfile = os.path.join(serving_dir, "Dockerfile")
    servepy    = os.path.join(serving_dir, "serve.py")
    if not os.path.exists(dockerfile):
        raise FileNotFoundError(
            f"❌ Dockerfile not found at '{dockerfile}'.\n"
            f"   It should have been created by create_serving_artifacts().\n"
            f"   Run: main.create_serving_artifacts('{serving_dir}', gcs_uri)"
        )
    print(f"   ✓ Dockerfile : {dockerfile}")
    print(f"   ✓ serve.py   : {servepy}")

    # ── STEP 1: Artifact Registry repo ───────────────────────
    print(f"\n📦 Creating Artifact Registry repo '{repo_name}' …")
    r = subprocess.run([
        "gcloud", "artifacts", "repositories", "create", repo_name,
        "--repository-format=docker",
        f"--location={region}",
        f"--project={project_id}",
        "--quiet",
    ], capture_output=True, text=True)
    msg = r.stderr.strip()
    print(f"   {'✓ Created.' if r.returncode == 0 else '✓ Already exists.' if 'already exists' in msg.lower() else '⚠️  ' + msg}")

    # ── STEP 2: Docker auth ───────────────────────────────────
    print("\n🔑 Configuring Docker auth for Artifact Registry …")
    rc = _run_streamed([
        "gcloud", "auth", "configure-docker",
        f"{region}-docker.pkg.dev", "--quiet",
    ])
    if rc != 0:
        raise RuntimeError("❌ Docker auth configuration failed. See output above.")
    print("   ✓ Docker auth OK.")

    # ── STEP 3: Cloud Build ───────────────────────────────────
    # os.system() is used here intentionally — it routes output
    # directly to the notebook cell, unlike subprocess which
    # Colab swallows when called from Python functions.
    print(f"\n🔨 Cloud Build starting (~5-10 min) …")
    print(f"   Image target : {image_uri}\n")
    print("─" * 60)

    build_cmd = (
        f"gcloud builds submit {serving_dir}"
        f" --tag={image_uri}"
        f" --project={project_id}"
        f" --machine-type=E2_HIGHCPU_8"
        f" --timeout=1200"
    )
    rc = os.system(build_cmd)

    print("─" * 60)

    if rc != 0:
        r = subprocess.run([
            "gcloud", "builds", "list", "--limit=1",
            "--format=value(id)", f"--project={project_id}",
        ], capture_output=True, text=True)
        build_id = r.stdout.strip()
        raise RuntimeError(
            f"\n❌ Cloud Build FAILED (exit code {rc})\n"
            f"   The full error is printed above this traceback.\n"
            f"   Build ID  : {build_id}\n"
            f"   Console   : https://console.cloud.google.com/cloud-build/builds"
            f"/{build_id}?project={project_id}\n\n"
            f"   Common fixes:\n"
            f"   • Billing not enabled  → console.cloud.google.com/billing\n"
            f"   • API not enabled      → !gcloud services enable cloudbuild.googleapis.com\n"
            f"   • Wrong project        → !gcloud config set project YOUR_PROJECT_ID\n"
            f"   • Quota exceeded       → change GCP_REGION to 'us-east1'"
        )

    print(f"\n✅ Image pushed → {image_uri}")
    return image_uri


def get_build_logs(build_id: str, project_id: str = GCP_PROJECT_ID) -> None:
    """
    Print Cloud Build logs for a failed build.
    Call this right after a Cloud Build failure to see the exact error.

    Example
    -------
    main.get_build_logs("abc123-your-build-id")
    """
    print(f"📋 Fetching logs for build {build_id} …\n")
    subprocess.run([
        "gcloud", "builds", "log", build_id,
        f"--project={project_id}",
    ])


# ────────────────────────────────────────────────────────────
#  STEP 4 — Register custom image model in Vertex AI
# ────────────────────────────────────────────────────────────

def register_vertex_model(
    image_uri:   str,
    gcs_uri:     str,
    model_name:  str = GCP_MODEL_NAME,
    project_id:  str = GCP_PROJECT_ID,
    region:      str = GCP_REGION,
):
    """
    Register the custom container image as a Vertex AI Model.
    No artifact_uri — avoids the .mar format validator entirely.

    Returns
    -------
    Vertex AI Model object

    Example
    -------
    vertex_model = main.register_vertex_model(image_uri, gcs_uri)
    """
    _, aiplatform = _gcp()
    aiplatform.init(project=project_id, location=region)

    print(f"📋 Registering '{model_name}' in Vertex AI …")

    # Pass GCS bucket + prefix as separate env vars.
    # serve.py uses google-cloud-storage to download the model at startup.
    # /gcs/ path mounting is NOT automatic for custom containers.
    gcs_clean  = gcs_uri.replace("gs://", "")
    gcs_bucket = gcs_clean.split("/")[0]
    gcs_prefix = "/".join(gcs_clean.split("/")[1:])

    vertex_model = aiplatform.Model.upload(
        display_name                = model_name,
        serving_container_image_uri = image_uri,
        serving_container_ports     = [8080],
        serving_container_health_route  = "/health",
        serving_container_predict_route = "/predict",
        serving_container_environment_variables = {
            "GCS_BUCKET": gcs_bucket,
            "GCS_PREFIX": gcs_prefix,
        },
    )

    print(f"✅ Registered → {vertex_model.resource_name}")
    return vertex_model


# ────────────────────────────────────────────────────────────
#  STEP 5 — Create endpoint
# ────────────────────────────────────────────────────────────

def create_endpoint(
    endpoint_name: str = GCP_ENDPOINT_NAME,
    project_id:    str = GCP_PROJECT_ID,
    region:        str = GCP_REGION,
):
    """
    Create a Vertex AI Endpoint (the URL that receives requests).

    Returns
    -------
    Vertex AI Endpoint object

    Example
    -------
    endpoint = main.create_endpoint()
    """
    _, aiplatform = _gcp()
    aiplatform.init(project=project_id, location=region)
    print(f"🌐 Creating endpoint '{endpoint_name}' …")
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f"✅ Endpoint created → {endpoint.resource_name}")
    return endpoint


# ────────────────────────────────────────────────────────────
#  STEP 6 — Deploy model to endpoint
# ────────────────────────────────────────────────────────────

def deploy_to_endpoint(
    vertex_model,
    endpoint,
    machine_type: str  = GCP_MACHINE_TYPE,
    gpu_type            = GCP_GPU_TYPE,    # None for CPU, "NVIDIA_L4" or "NVIDIA_TESLA_T4" for GPU
    gpu_count:    int  = GCP_GPU_COUNT,    # 0 for CPU, 1 for GPU
    min_replicas: int  = 1,
    max_replicas: int  = 1,
) -> object:
    """
    Attach the model to the endpoint. Billing starts here.

    CPU deployment (always works — no quota needed):
        machine_type="n1-standard-8", gpu_type=None, gpu_count=0

    GPU deployments (requires quota approval):
        L4  → machine_type="g2-standard-4", gpu_type="NVIDIA_L4",        gpu_count=1
        T4  → machine_type="n1-standard-4", gpu_type="NVIDIA_TESLA_T4",  gpu_count=1

    Returns deployed model resource.

    Example
    -------
    # CPU
    deployed = main.deploy_to_endpoint(vertex_model, endpoint)

    # L4 GPU
    deployed = main.deploy_to_endpoint(vertex_model, endpoint,
        machine_type="g2-standard-4", gpu_type="NVIDIA_L4", gpu_count=1)
    """
    use_gpu = gpu_type is not None and gpu_count > 0

    if use_gpu:
        print(f"🚀 Deploying on GPU ({machine_type} + {gpu_type} ×{gpu_count}) …")
    else:
        print(f"🚀 Deploying on CPU ({machine_type}) …")
    print("   Takes 5–10 min …")

    deploy_kwargs = dict(
        endpoint           = endpoint,
        machine_type       = machine_type,
        min_replica_count  = min_replicas,
        max_replica_count  = max_replicas,
        traffic_percentage = 100,
    )

    # Only pass accelerator args when using GPU
    # Passing accelerator_type=None to Vertex AI raises a ValueError
    if use_gpu:
        deploy_kwargs["accelerator_type"]  = gpu_type
        deploy_kwargs["accelerator_count"] = gpu_count

    deployed = vertex_model.deploy(**deploy_kwargs)
    print("✅ Deployed!  ⚠️  Billing is now running.")
    return deployed


# ────────────────────────────────────────────────────────────
#  STEP 7 — Send predictions
# ────────────────────────────────────────────────────────────

def predict_gcp(
    endpoint,
    question:          str,
    max_new_tokens:    int   = MAX_NEW_TOKENS,
    temperature:       float = TEMPERATURE,
) -> str:
    """
    Send a question to the live Vertex AI endpoint and return the answer.

    Uses the Vertex AI custom container request format:
      {"instances": [{"inputs": "...", "parameters": {...}}]}

    Returns
    -------
    answer (str)

    Example
    -------
    answer = main.predict_gcp(endpoint, "What is the 50/30/20 rule?")
    print(answer)
    """
    instances = [{
        "inputs": question,
        "parameters": {
            "max_new_tokens":     max_new_tokens,
            "temperature":        temperature,
            "repetition_penalty": REPETITION_PENALTY,
        },
    }]

    response = endpoint.predict(instances=instances)
    pred     = response.predictions[0]

    if isinstance(pred, list):
        answer = pred[0].get("generated_text", str(pred[0]))
    elif isinstance(pred, dict):
        answer = pred.get("generated_text", str(pred))
    else:
        answer = str(pred)

    return answer.strip()


def run_gcp_test_predictions(endpoint) -> None:
    """
    Run 5 test questions against the live endpoint.

    Example
    -------
    main.run_gcp_test_predictions(endpoint)
    """
    questions = [
        "What is the 50/30/20 rule?",
        "How does compound interest work?",
        "What is a good credit score?",
        "Should I pay off debt or invest first?",
        "What is an index fund?",
    ]
    print("🧪 Testing live endpoint …\n")
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {predict_gcp(endpoint, q)}")
        print("-" * 60)
    print("✅ All test predictions done.")


# ────────────────────────────────────────────────────────────
#  STEP 8 — Teardown (stops billing)
# ────────────────────────────────────────────────────────────

def undeploy_and_delete(
    endpoint,
    vertex_model,
    delete_model:    bool = True,
    delete_endpoint: bool = True,
) -> None:
    """
    Undeploy model and delete endpoint to stop billing immediately.
    ⚠️  GCP charges ~$7/day minimum while any endpoint is live.
       Always call this when finished testing.

    Example
    -------
    main.undeploy_and_delete(endpoint, vertex_model)
    """
    print("🗑  Tearing down GCP resources …")
    endpoint.undeploy_all()
    if delete_endpoint:
        endpoint.delete()
        print("   ✓ Endpoint deleted.")
    if delete_model:
        vertex_model.delete()
        print("   ✓ Model deleted from registry.")
    print("✅ Teardown complete. Billing stopped.")


# ────────────────────────────────────────────────────────────
#  Cost estimate helper
# ────────────────────────────────────────────────────────────

def estimate_gcp_cost(hours: float = 1.0) -> None:
    """
    Print an estimated cost before deploying.

    Example
    -------
    main.estimate_gcp_cost(hours=2.0)
    """
    t4_rate = 0.54   # n1-standard-4 + T4  (us-central1, approx)
    l4_rate = 0.80   # n1-standard-4 + L4
    print(f"\n💰 Estimated Vertex AI cost for {hours:.1f} hour(s):")
    print(f"   T4 (n1-standard-4) : ${t4_rate * hours:.2f}")
    print(f"   L4 (n1-standard-4) : ${l4_rate * hours:.2f}")
    print(f"   ⚠️  Minimum ~$7/day while endpoint is live.")
    print(f"   ✅ Always call undeploy_and_delete() when done.")


# ────────────────────────────────────────────────────────────
#  Full pipeline — one call does everything
# ────────────────────────────────────────────────────────────

def deploy_pipeline_gcp(
    local_model_dir: str  = MERGED_PATH,
    serving_dir:     str  = "./serving",
    repo_name:       str  = "phi3-finance-repo",
    image_name:      str  = "phi3-finance-server",
    gcs_prefix:      str  = "models/phi3-finance",
    machine_type:    str  = GCP_MACHINE_TYPE,
    gpu_type               = GCP_GPU_TYPE,    # None for CPU
    gpu_count:       int  = GCP_GPU_COUNT,    # 0 for CPU
    run_tests:       bool = True,
) -> dict:
    """
    One call runs the full GCP deployment pipeline:
        upload weights → write serving files → build & push image
        → register model → create endpoint → deploy → test

    CPU (default, always works):
        deploy_pipeline_gcp()

    GPU (requires quota):
        deploy_pipeline_gcp(
            machine_type = "g2-standard-4",
            gpu_type     = "NVIDIA_L4",
            gpu_count    = 1,
        )

    Returns {"gcs_uri", "image_uri", "vertex_model", "endpoint", "deployed_model"}
    Call undeploy_and_delete() on returned resources when done.

    Example
    -------
    resources = main.deploy_pipeline_gcp()
    answer    = main.predict_gcp(resources["endpoint"], "What is a FICO score?")
    main.undeploy_and_delete(resources["endpoint"], resources["vertex_model"])
    """
    use_gpu    = gpu_type is not None and gpu_count > 0
    cpu_mode   = not use_gpu
    mode_label = f"GPU ({gpu_type})" if use_gpu else "CPU (no GPU)"

    estimate_gcp_cost(hours=1.0)
    print(f"\n   Deployment mode: {mode_label}")
    confirm = input("\n⚠️  This starts billing. Continue? (yes/no): ").strip()
    if confirm.lower() != "yes":
        print("Cancelled.")
        return {}

    print("\n" + "="*60)
    print(f" GCP VERTEX AI DEPLOYMENT — {mode_label.upper()}")
    print("="*60)

    # 1. Upload weights
    gcs_uri = upload_model_to_gcs(
        local_model_dir = local_model_dir,
        gcs_prefix      = gcs_prefix,
    )

    # 2. Write serve.py + Dockerfile (cpu_mode controls model loading + base image)
    create_serving_artifacts(serving_dir, gcs_uri, cpu_mode=cpu_mode)

    # 3. Build & push image
    image_uri = build_and_push_image(
        serving_dir = serving_dir,
        repo_name   = repo_name,
        image_name  = image_name,
    )

    # 4. Register model
    vertex_model = register_vertex_model(image_uri, gcs_uri)

    # 5. Create endpoint
    endpoint = create_endpoint()

    # 6. Deploy
    deployed = deploy_to_endpoint(
        vertex_model,
        endpoint,
        machine_type = machine_type,
        gpu_type     = gpu_type,
        gpu_count    = gpu_count,
    )

    # 7. Test
    if run_tests:
        run_gcp_test_predictions(endpoint)

    print("\n" + "="*60)
    print(" DEPLOYMENT COMPLETE")
    print(f" Mode     : {mode_label}")
    print(f" Endpoint : {endpoint.resource_name}")
    print(" ⚠️  Call undeploy_and_delete() when done to stop billing!")
    print("="*60)

    return {
        "gcs_uri":        gcs_uri,
        "image_uri":      image_uri,
        "vertex_model":   vertex_model,
        "endpoint":       endpoint,
        "deployed_model": deployed,
    }
