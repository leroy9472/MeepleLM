<div align="center">

# 🎲 MeepleLM
*A Virtual Playtester Simulating Diverse Subjective Experiences in Board Games*

</div>

<p align="center">
  <img src="./assets/overview.pdf" alt="MeepleLM Framework Overview" width="850"/>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <a href="https://github.com/hiyouga/LLaMA-Factory"><img alt="Training" src="https://img.shields.io/badge/Training-LLaMA--Factory-orange"></a>
  <a href="https://docs.vllm.ai/en/latest/"><img alt="Inference" src="https://img.shields.io/badge/Inference-vLLM-blueviolet"></a>
  <a href="https://arxiv.org/abs/2601.07251"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2601.07251-red"></a>
</p>

---

## 📖 Table of Contents
- [📜 Abstract](#-abstract)
- [📂 File Structure](#-file-structure)
- [💾 Datasets](#-datasets)
- [🤖 Models & Checkpoints](#-models--checkpoints)
- [🚀 Training](#-training)
- [⚡ Inference & Evaluation](#-inference--evaluation)
- [📄 Citation](#-citation)

---

### 📜 Abstract

> Recent advancements have expanded the role of Large Language Models in board games from playing agents to creative co-designers. However, a critical gap remains: current systems lack the capacity to offer constructive critique grounded in the emergent user experience. To bridge this gap, we introduce **MeepleLM**, a virtual playtester that simulates diverse subjective experiences. We curated a dataset of **1,727 structurally corrected rulebooks** and **150K reviews** selected via quality scoring. We augment this data with **Mechanics-Dynamics-Aesthetics (MDA)** reasoning chains to explicitly recover the latent dynamics connecting written rules to player satisfaction. Furthermore, we distill five distinct player personas (e.g., *System Purist*, *Social Lubricator*) to model subjective heterogeneity.

---

### 📂 File Structure

```text
.
├── assets/                    # Project images and figures
├── data/
│   ├── metadata/              # Meta-info (Game IDs, names, BGG stats, splits)
│   ├── finetuning/            # Alpaca-formatted datasets for training/testing
│   ├── reviews/               # Raw and filtered review data
│   └── rulebooks/             # Structured Markdown rulebooks
├── checkpoints/               # LoRA adapters for MeepleLM & Ablations
├── training/                  # YAML configurations for LLaMA-Factory
├── inference/                 # Inference scripts (vLLM example)
└── results/                   # Generated critiques and evaluation outputs

```

---

### 💾 Datasets

We provide the complete pipeline data, from raw sources to instruction-tuning ready files.

* **`data/metadata/`**:
* `game_info.json`: Mappings of Game ID to metadata (Name, Rank, Weight, Year).
* `test_games_list.json`: The official evaluation split (207 games) used in the paper.


* **`data/finetuning/`**: Ready-to-use **Alpaca format** datasets for SFT. Each folder contains `_train.json` and `_test.json`.
* `MeepleLM/`: Full dataset with MDA CoT reasoning chains.
* `wo_MDA/`: Ablation without reasoning chains (Direct generation).
* `wo_Persona/`: Ablation without persona profiles.
* `wo_Rulebook/`: Ablation without rule context (Parametric knowledge only).


* **`data/rulebooks/`**: The corpus of 1,727 processed rulebooks in Markdown format.
* **`data/reviews/`**: The filtered high-quality review corpus used to construct the training data.

---

### 🤖 Models & Checkpoints

We provide **LoRA adapters** trained on **Qwen2.5-7B-Instruct** (referred to as Qwen3-8B in the paper context). These can be loaded easily using [vLLM](https://docs.vllm.ai/).

| Model Variant | Description | Path |
| --- | --- | --- |
| **MeepleLM (Ours)** | Full model with Persona-conditioning and MDA reasoning. | `./checkpoints/MeepleLM/` |
| **w/o MDA** | Ablation removing Chain-of-Thought reasoning. | `./checkpoints/wo_MDA/` |
| **w/o Persona** | Ablation using a generic player prompt. | `./checkpoints/wo_Persona/` |
| **w/o Rulebook** | Ablation relying solely on internal knowledge. | `./checkpoints/wo_Rulebook/` |

#### Serving with vLLM

You can serve the model with the LoRA adapter enabled. For example, to serve MeepleLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules MeepleLM=checkpoints/MeepleLM \
    --served-model-name MeepleLM \
    --port 8000

```

---

### 🚀 Training

All models were trained using the **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** framework. We provide the exact YAML configurations used for our experiments in the `training/` directory.

To reproduce the training process:

1. **Install LLaMA-Factory:**
Please refer to the [official LLaMA-Factory repository](https://github.com/hiyouga/LLaMA-Factory) for installation instructions.
2. **Register Datasets:**
Add the paths from `data/finetuning/` to LLaMA-Factory's `data/dataset_info.json`.
3. **Run Training:**
```bash
llamafactory-cli train training/train_meeplelm.yaml

```


*(Note: Config files for ablation studies are also provided in the `training/` folder.)*

---

### ⚡ Inference & Evaluation

The `inference/` directory contains scripts to generate virtual playtest reports.

* **`playtest_inference.py`**: A sample script designed to work with the **MeepleLM** checkpoint served via vLLM. It iterates through the test set games, applying the Persona constraints to generate reviews.
* **`results/`**: Stores the output JSON files generated by the model (e.g., `results/inference_meeplelm/`).

> **Note:** The provided inference script is configured for the **MeepleLM** LoRA adapter and local vLLM server. If you wish to evaluate other models (e.g., GPT-4o, Claude) or use different API endpoints, please modify the `API_URL` and `MODEL_NAME` parameters in the script accordingly.

---

### 📄 Citation

If you use MeepleLM, the rulebook dataset, or the persona taxonomy in your research, please cite our paper:

```bibtex
@article{li2026meeplelm,
  title={MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences},
  author={Li, Zizhen and Li, Chuanhao and Wang, Yibin and Feng, Yukang and Sun, Jianwen and Ai, Jiaxin and Zhang, Fanrui and Sun, Mingzhu and Huang, Yifei and Zhang, Kaipeng},
  journal={arXiv preprint arXiv:2601.07251},
  year={2026}
}
