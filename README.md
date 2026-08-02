# Phi-3 Mini Personal Finance Assistant

Fine-tune Microsoft's **Phi-3 Mini (3.8B)** on personal finance Q&A data using QLoRA,
evaluate it, merge the adapter, launch a Gradio chat UI, and deploy to **GCP Vertex AI**.

---

##  Repository Structure

```
phi3-finance/
├── main.py                              ← Full pipeline with GCP deployment
├── main_no_gcp.py                       ← Training + Gradio UI only (no GCP)
├── phi3_finance_caller.ipynb            ← Notebook for main.py (Sections 1–10, GCP)
├── phi3_finance_caller_no_gcp.ipynb     ← Notebook for main_no_gcp.py (Sections 1–9)
├── phi3_finance_huggingface.ipynb       ← Hugging Face deployment (Section 11)
├── personal_finance_qa.csv              ← Your training data (400+ Q&A pairs) ← YOU PROVIDE
├── docs/
│   ├── gcp_deployment_debugging_log.pdf ← Full record of every error + fix
│   └── huggingface_deployment_guide.pdf ← Step-by-step HF guide
├── requirements.txt
├── .gitignore
└── README.md
```

> **Which files to use?**
> - Just training + Gradio UI → `main_no_gcp.py` + `phi3_finance_caller_no_gcp.ipynb`
> - Full pipeline with cloud deployment → `main.py` + `phi3_finance_caller.ipynb`

---

## Quick Start (Google Colab)

1. Open the notebook in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU (A100 or T4)**: `Runtime → Change runtime type → GPU`
3. Upload `main.py` (or `main_no_gcp.py`) and `personal_finance_qa.csv` to Colab
4. Run the session start cell to import, then run top to bottom

---

## Pipeline Overview

| Section | What happens | Est. time | In no-GCP version |
|---|---|---|---|
| **1 – Setup** | GPU check, install libraries | 5 min | ✅ |
| **2 – Model** | Load Phi-3 Mini in 4-bit + apply LoRA adapters | 2 min | ✅ |
| **3 – Dataset** | Load CSV, format into Phi-3 chat template | 1 min | ✅ |
| **4 – Train v1** | Fine-tune for 3 epochs | 30–60 min | ✅ |
| **5 – Inference** | Single + multi-turn chat | instant | ✅ |
| **6 – Evaluate** | 30-prompt evaluation, manual 0–2 scoring | 15 min | ✅ |
| **7 – Retrain v2** | Fix failed questions, retrain on improved data | 30–45 min | ✅ |
| **8 – Merge** | Merge LoRA adapter into base model | 5 min | ✅ |
| **9 – Gradio UI** | Launch shareable chat UI (public link 72hrs) | 1 min | ✅ |
| **10 – GCP** | Deploy to Vertex AI (CPU or GPU) | 15–20 min | ❌ |

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

## Model Details

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

## GCP Deployment (Section 10)

### Authentication in Colab (required every session)

```python
# Always run both at the start of every Colab session
from google.colab import auth
auth.authenticate_user()             # covers Python SDK (google-cloud-*)
!gcloud config set project YOUR_PROJECT_ID  # covers gcloud CLI commands
```

> `gcloud auth application-default login` **crashes in Colab** — use
> `google.colab.auth.authenticate_user()` instead for Python SDK authentication.

### One-Time Setup

```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com

# Grant IAM permissions (replace PROJECT_NUMBER with yours)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

### Deployment — CPU (Recommended — No Quota Needed)

```python
# CPU deployment — always works, no GPU quota required
resources = main.deploy_pipeline_gcp()

# Test it
answer = main.predict_gcp(resources["endpoint"], "What is the 50/30/20 rule?")

# Always teardown when done — GCP charges ~$7/day minimum while live
main.undeploy_and_delete(resources["endpoint"], resources["vertex_model"])
```

### Deployment — GPU (Requires Quota Approval)

```python
# L4 GPU — fastest option for Phi-3 Mini
resources = main.deploy_pipeline_gcp(
    machine_type = "g2-standard-4",
    gpu_type     = "NVIDIA_L4",       # correct name (not NVIDIA_TESLA_L4)
    gpu_count    = 1,
)

# T4 GPU — cheaper alternative
resources = main.deploy_pipeline_gcp(
    machine_type = "n1-standard-4",
    gpu_type     = "NVIDIA_TESLA_T4",
    gpu_count    = 1,
)
```

### Deployment Flow

```
Auth (every session)
      ↓
Upload weights → GCS bucket
      ↓
Write serve.py + Dockerfile  (CPU: python:3.10-slim | GPU: pytorch/pytorch:2.3.0-cuda12.1)
      ↓
Build & push Docker image  (Cloud Build → Artifact Registry)
      ↓
Register model in Vertex AI  (no artifact_uri — avoids .mar validator)
      ↓
Create endpoint
      ↓
Deploy  (CPU: n1-standard-8 | GPU: g2-standard-4 + NVIDIA_L4)
      ↓
Test predictions
      ↓
Teardown — stops billing
```

### GCP Cost Reference

| Option | Machine | Cost/hr | Response time | Quota needed |
|---|---|---|---|---|
| **CPU** ← recommended | n1-standard-8 | ~$0.38 | ~20-30s | ❌ None |
| T4 GPU | n1-standard-4 | ~$0.54 | ~2s | ✅ Must request |
| L4 GPU | g2-standard-4 | ~$0.80 | ~1s | ✅ Must request |

> GCP charges ~$7/day minimum while any endpoint is live — even with zero traffic.
> Always call `undeploy_and_delete()` when finished.

### GPU Quota

New GCP projects start with **zero GPU quota** for Vertex AI serving.
To request quota:
1. Go to `console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas`
2. Search `Custom model serving T4` or `Custom model serving L4`
3. Region: `us-central1`
4. Request limit: `1`

**Use CPU deployment while waiting** — the API is identical, just slower responses.

### Reconnecting After Colab Restart

Variables like `vertex_model` and `endpoint` are lost on restart.
Reconnect to existing GCP resources:

```python
from google.cloud import aiplatform
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# List existing resources
for m in aiplatform.Model.list():
    print(m.display_name, m.resource_name)
for e in aiplatform.Endpoint.list():
    print(e.display_name, e.resource_name)

# Reconnect by resource name
vertex_model = aiplatform.Model("projects/.../locations/.../models/MODEL_ID")
endpoint     = aiplatform.Endpoint("projects/.../locations/.../endpoints/ENDPOINT_ID")
```

---

## Evaluation Scoring Guide

After training, 30 unseen prompts are tested across 6 topic areas:

| Score | Meaning | Action |
|---|---|---|
| `2` | Correct, clear, complete | Keep — model learned this well |
| `1` | Mostly correct but missing detail | Accept |
| `0` | Wrong, hallucinated, or refused | Fix — add better training data |

**Target:** average ≥ 1.5 (75%) before merging.
If below 75%, add fixes in Section 7 and retrain.

---

## Key Learnings

- **QLoRA** fits a 3.8B model in ~6 GB VRAM using 4-bit quantization — no expensive hardware required
- **LoRA rank 16** trains <2% of parameters but meaningfully changes model behaviour
- **Data quality > quantity** — fixing 5 bad rows beats adding 50 new ones
- **Two GCP auths in Colab** — `auth.authenticate_user()` for Python SDK; `gcloud auth login` for CLI
- **`gcloud auth application-default login` crashes in Colab** — use `google.colab.auth` instead
- **New GCP projects have zero GPU quota** — always build a CPU fallback path
- **Custom Docker containers** are more reliable than pre-built DLC/TGI images on Vertex AI
- **Region consistency is mandatory** — model, endpoint, and Artifact Registry must all be the same region
- **Colab runtime restarts wipe everything** — `/content/` folder, Python variables, and gcloud auth all reset

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Unsloth](https://github.com/unslothai/unsloth) | Fast LoRA fine-tuning |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter management |
| [TRL](https://github.com/huggingface/trl) | SFTTrainer |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization |
| [Gradio](https://gradio.app) | Chat UI |
| [GCP Vertex AI](https://cloud.google.com/vertex-ai) | Cloud deployment |
| [Cloud Build](https://cloud.google.com/build) | Docker image builder (no local Docker needed) |
| [Artifact Registry](https://cloud.google.com/artifact-registry) | Docker image storage |
| [FastAPI + Uvicorn](https://fastapi.tiangolo.com) | Custom serving container |

---

## GCP Vertex AI Deployment — What Actually Made It Work

Deploying a 7.6 GB merged model to Vertex AI via a **custom Docker container** took several
iterations. These are the fixes that matter — bake them all in or the container fails silently:

| # | Problem | Fix |
|---|---|---|
| 1 | Pre-built DLC/TGI images expect `.mar` format | Use a **custom FastAPI container**; drop `artifact_uri` from `Model.upload()` |
| 2 | `/gcs/` mount path doesn't exist in custom containers | **Download the model from GCS** at startup via the `google-cloud-storage` SDK |
| 3 | Container passes health checks but never loads model (silent restart loop) | **Load the model in a background thread** (`async def startup` + `run_in_executor`) so `/health` returns 200 instantly |
| 4 | `Tokenizer class TokenizersBackend does not exist` | Strip `"tokenizer_class"` from `tokenizer_config.json` at startup; fall back to `LlamaTokenizer` |
| 5 | `Trying to set a tensor of shape … looks incorrect` | **Delete the duplicate `model.safetensors`** — the folder had both a sharded model and a single-file model, causing a weight collision |
| 6 | Out of memory loading on CPU | `low_cpu_mem_usage=True` + `n1-highmem-8` (52 GB RAM) |
| 7 | `DTensor` import error | `python:3.11-slim` base + pin `transformers==4.46.3`, `torch==2.4.1` |
| 8 | Invisible container logs | `PYTHONUNBUFFERED=1` + explicit `sys.stdout.flush()` after every log line |

### Serving container config (working)

```
Base image     : python:3.11-slim
Pinned deps    : torch==2.4.1, transformers==4.46.3, accelerate==0.34.2,
                 sentencepiece, protobuf, fastapi, uvicorn, google-cloud-storage
Model source   : downloaded from GCS at startup (not /gcs/ mount)
Tokenizer      : tokenizer_class stripped + use_fast=False / LlamaTokenizer fallback
Model load     : torch.float32, low_cpu_mem_usage=True
Startup        : async background thread so /health responds immediately
Machine        : n1-highmem-8 (CPU, no GPU quota needed)
Deploy timeout : deploy_request_timeout=1800
```

### One-time IAM setup

The following role grants are required (replace `PROJECT_NUMBER`):

```bash
# Cloud Build
roles/storage.objectAdmin, roles/cloudbuild.builds.builder   → PROJECT_NUMBER@cloudbuild.gserviceaccount.com

# Compute service account
roles/artifactregistry.writer, roles/artifactregistry.reader,
roles/storage.objectAdmin, roles/logging.logWriter           → PROJECT_NUMBER-compute@developer.gserviceaccount.com

# Vertex AI service agent (needed to PULL the image)
roles/artifactregistry.reader                                → service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com
```

> **Billing:** a live Vertex AI endpoint costs ~$7/day even with zero traffic.
> Always run the teardown cell (`endpoint.undeploy_all(); endpoint.delete(); vertex_model.delete()`) when done.

> **Auth in Colab:** use `google.colab.auth.authenticate_user()` — `gcloud auth application-default login` **crashes** in Colab.

---

## Simpler Alternative — Hugging Face

GCP Vertex AI gives full infrastructure control but is complex. For a **free, public demo**
(no Docker, no IAM, no quota, $0 while idle), deploy the same model to Hugging Face instead:

- **Model Hub** — push the merged model; loadable in one line, with an automatic inference widget
- **Spaces** — host a live Gradio chat app with a public URL (~20 minutes)

See `phi3_finance_huggingface.ipynb` for the step-by-step notebook. The same two fixes apply:
`use_fast=False` on the tokenizer and `low_cpu_mem_usage=True` on the model.

| | GCP Vertex AI | Hugging Face Spaces |
|---|---|---|
| Setup effort | High (containers, IAM, quota) | Low (two files) |
| Public URL | Manual wiring | Automatic |
| UI included | No (API only) | Yes (Gradio) |
| Cost when idle | ~$7/day | $0 |
| Best for | Enterprise / private / scale | Portfolio / demo / sharing |
