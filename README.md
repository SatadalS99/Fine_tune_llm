# 💰 Phi-3 Mini Personal Finance Assistant

Fine-tune Microsoft's **Phi-3 Mini (3.8B)** on personal finance Q&A data using QLoRA,
evaluate it, merge the adapter, launch a Gradio chat UI, and deploy to **GCP Vertex AI**.

---

## 📁 Repository Structure

```
phi3-finance/
├── main.py                        ← All functions (import this in the notebook)
├── phi3_finance_caller.ipynb      ← Notebook — calls functions from main.py
├── personal_finance_qa.csv        ← Your training data (400+ Q&A pairs) ← YOU PROVIDE
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

1. Open `phi3_finance_caller.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU (A100 or T4)**: `Runtime → Change runtime type → GPU`
3. Upload `main.py` and `personal_finance_qa.csv` to Colab
4. Run cells top to bottom

---

## 📊 Pipeline Overview

| Section | What happens | Est. time |
|---|---|---|
| **1 – Setup** | GPU check, install libraries | 5 min |
| **2 – Model** | Load Phi-3 Mini in 4-bit + apply LoRA adapters | 2 min |
| **3 – Dataset** | Load CSV, format into Phi-3 chat template | 1 min |
| **4 – Train v1** | Fine-tune for 3 epochs | 30–60 min |
| **5 – Inference** | Single + multi-turn chat | instant |
| **6 – Evaluate** | 30-prompt evaluation, manual 0–2 scoring | 15 min |
| **7 – Retrain v2** | Fix failed questions, retrain on improved data | 30–45 min |
| **8 – Merge** | Merge LoRA adapter into base model | 5 min |
| **9 – Gradio UI** | Launch shareable chat UI (public link 72hrs) | 1 min |
| **10 – GCP** | Deploy to Vertex AI (CPU or GPU) | 15–20 min |

---

## 🗂 Dataset Format

Your CSV must have exactly these two columns:

```csv
Question,Answer
What is the 50/30/20 rule?,The 50/30/20 rule splits your after-tax income into...
How does compound interest work?,Compound interest is interest calculated on both...
```

Recommended: **400–500 rows** covering 6–8 personal finance topics.

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Base model | `unsloth/Phi-3-mini-4k-instruct` |
| Parameters | 3.8 billion |
| Quantization | 4-bit (QLoRA) |
| LoRA rank | 16 |
| Context length | 4,096 tokens |
| Training method | SFTTrainer (TRL) |
| Fine-tuning topic | Personal finance Q&A |

---

## ☁️ GCP Deployment

### Prerequisites

```bash
# Install and authenticate gcloud CLI
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com
```

### Deployment Flow

```
Upload weights to GCS
      ↓
Build custom Docker image (Cloud Build)
      ↓
Register model in Vertex AI
      ↓
Create endpoint
      ↓
Deploy (GPU: g2-standard-4 + NVIDIA_L4 | CPU: n1-standard-8)
      ↓
Test predictions
      ↓
⚠️  Teardown when done (stops billing)
```

### GCP Cost Reference

| Option | Machine | Cost/hr | Notes |
|---|---|---|---|
| L4 GPU | g2-standard-4 | ~$0.80/hr | Fastest, needs quota |
| T4 GPU | n1-standard-4 | ~$0.54/hr | Good balance, needs quota |
| CPU | n1-standard-8 | ~$0.38/hr | Always available, ~20-30s/response |

> ⚠️ **Important:** GCP charges ~$7/day minimum while any endpoint is live.
> Always delete the endpoint when finished.

### GPU Quota

New GCP projects start with **zero GPU quota** for Vertex AI.
To request quota:
1. Go to `console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas`
2. Search `Custom model serving`
3. Request 1 GPU for `us-central1`

Use **CPU deployment** while waiting for quota approval.

---

## 💬 Evaluation Scoring Guide

After training, 30 unseen prompts are tested:

| Score | Meaning |
|---|---|
| `2` | Correct, clear, complete |
| `1` | Mostly correct but missing detail |
| `0` | Wrong, hallucinated, or refused |

**Target:** average ≥ 1.5 (75%) before merging.
If below 75%, add fixes in Section 7 and retrain.

---

## 🔑 Key Learnings

- **QLoRA** fits a 3.8B model in ~6GB VRAM using 4-bit quantization
- **LoRA rank 16** trains <2% of parameters but meaningfully changes behavior
- **Data quality > quantity** — fixing 5 bad rows beats adding 50 new ones
- **Two GCP auths needed in Colab** — `auth.authenticate_user()` for Python SDK, `gcloud auth login` for CLI
- **GPU quota** — new projects start at 0; CPU deployment always works as fallback
- **Region consistency** — model, endpoint, and image registry must all be in the same GCP region

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| [Unsloth](https://github.com/unslothai/unsloth) | Fast LoRA fine-tuning |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter management |
| [TRL](https://github.com/huggingface/trl) | SFTTrainer |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization |
| [Gradio](https://gradio.app) | Chat UI |
| [GCP Vertex AI](https://cloud.google.com/vertex-ai) | Cloud deployment |
| [Cloud Build](https://cloud.google.com/build) | Docker image builder |
| [Artifact Registry](https://cloud.google.com/artifact-registry) | Docker image storage |
