# Aletheia: A Medical Question-Answering RAG System

CSCI E-222 final project, Harvard Extension School, Spring 2026.
Author: Fady A Eid.

A retrieval-augmented question-answering pipeline over MedQuAD (16,406 NIH medical Q&A pairs). MedCPT bi-encoder retrieves the top-3 passages per question; BioMistral-7B-SLERP generates the answer under greedy decoding. Evaluated with Top-k Accuracy, MRR, ROUGE-L F1, and BERTScore F1, plus a 50-question RAG-vs-no-RAG ablation.

See `aletheia_report.pdf` for the full write-up, methods, results, and references. The video walkthrough URL is on the report cover and on the last slide.

## Environment

Built and tested on:

- Hardware: Dell Pro Max with NVIDIA GB10 Grace Blackwell Superchip, 128 GB unified memory, aarch64.
- OS: DGX OS, CUDA 13.0.
- Python: 3.12.

Reproduces on Google Colab with the caveats in report Section 4.5 (Colab Pro: A100 or L4, BF16-native; free-tier T4 requires 4-bit quantization via `bitsandbytes`).

## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The `--extra-index-url` flag tells pip to fall back to PyPI for non-cu130 packages while pulling the CUDA 13.0 build of `torch`. On Colab x86_64 you can omit the flag and use the default PyPI index.

## Data

The notebook loads MedQuAD from the Hugging Face `lavita/MedQuAD` mirror via `load_dataset` (handled in Section 2.1, no manual download needed). No separate download step is required.

A 200-row sample is included in `data/medquad_sample_200.csv` for quick inspection without triggering the full HF download.

## Pre-computed embeddings (optional, saves ~8 minutes)

The embedding step (Section 5 of the notebook) produces a 63 MB `embeddings.npy` of shape `(20604, 768)` float32. The notebook does not auto-detect the cached file. To skip the re-embedding step manually:

1. Download the pre-computed file from the link below and place it at `data/embeddings.npy`.
2. In cell 45 of the notebook, replace the embedding loop with:
   ```python
   import numpy as np
   embeddings = np.load(DATA_DIR / "embeddings.npy")
   assert embeddings.shape == (20604, 768)
   assert embeddings.dtype == np.float32
   ```
3. Run the rest of the notebook normally.

Pre-computed embeddings download: [URL added after GitHub Release upload]

## Reproducing the experiment

1. Install dependencies (above).
2. Open `aletheia.ipynb` in Jupyter.
3. Run cells top to bottom. The notebook covers: setup, data loading and preprocessing, exploratory analysis, chunking, embedding, FAISS indexing, retrieval and generation, evaluation, and visualization plus ablation.
4. End-to-end runtime on the GB10 is 90 to 140 minutes, dominated by RAG generation over the 1,641-question test split (about 80 minutes at 3 seconds per question). On Colab A100 the runtime is 60 to 90 minutes.

All seeds are fixed (`SEED = 42`) and `torch.use_deterministic_algorithms(True)` is set, so retrieval and generation are bit-exact reproducible on the same hardware.

## Submission package contents (this ZIP)

- `aletheia.ipynb`: the main notebook (entry point).
- `requirements.txt`: versioned dependencies (this file's neighbor).
- `README.md`: this file.
- `data/medquad_sample_200.csv`: 200-row sample of MedQuAD for inspection.

Not included (sizes exceed the 10 MB Canvas limit, hosted externally):

- `embeddings.npy` (63 MB): see the pre-computed embeddings link above.
- BioMistral-7B-SLERP weights (14.48 GB): downloaded by the notebook from Hugging Face on first run.
- Full MedQuAD (handled by `load_dataset` at runtime).

The report and slides are uploaded as separate files alongside this ZIP per the rubric.

## Citation

If you use this work, please cite the report and the four underlying systems it builds on: MedQuAD (Ben Abacha & Demner-Fushman, 2019), MedCPT (Jin et al., 2023), BioMistral (Labrak et al., 2024), and Lost-in-the-Middle (Liu et al., 2024). Full references are in `aletheia_report.pdf`.
