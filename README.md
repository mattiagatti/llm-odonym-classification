# 🛣️ Street Dedication Classification Pipeline

This project builds a normalized street database from Italian odonym datasets, classifies street dedications using an LLM-backed classifier, and produces thematic analyses and maps with a focus on gender representation.

> 📄 **Related paper:** *Revealing Gendered Patterns in Italy's Street Names with Large Language Models.* Published in [Journal of Computational Social Science (2026)](https://doi.org/10.1007/s42001-026-00486-z). See [Citation](#-citation) below.
---

## ⚙️ Environment Setup

### 1. Install Python and create a virtual environment

Tested with **Python 3.11**.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Hugging Face access token

Downloading the LLM weights requires a Hugging Face access token. The default model is
[`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct),
which is **gated**: you must first request and be granted access on its model page.

Create a token at:
https://huggingface.co/settings/tokens
(the `read` scope is sufficient)

Save the token, on a single line, in a file named:

```text
huggingface_api_key.txt
```

and place it in the project root directory. This file is read by `serve_llm.sh`.

> **Hardware note:** the default 70B model requires a multi-GPU setup with substantial VRAM.
> Use a smaller instruct model (via `--model`, see below) if your hardware is more limited.

### 4. Administrative boundaries (already included in the repository)

The official ISTAT boundary shapefiles used for thematic maps are already committed under
`input/italy_admin_boundaries/`, so **no action is needed** for a normal run.

Only if you want to re-download or refresh them, run:

```bash
mkdir -p input/italy_admin_boundaries && \
curl -L -o input/italy_admin_boundaries/Limiti01012025_g.zip \
  https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2025/Limiti01012025_g.zip && \
unzip -q input/italy_admin_boundaries/Limiti01012025_g.zip -d input/italy_admin_boundaries && \
rm input/italy_admin_boundaries/Limiti01012025_g.zip
```

---

## 🚀 Run the Pipeline

### 1. Data preprocessing

Prepare and enrich the street CSV by joining municipality metadata and normalizing fields.

```bash
python prepare_streets_dataset.py
```

Inputs (read from the `raw/` folder):
- `raw/STRAD_ITA_20251010.csv` — the street registry
- `raw/Elenco-comuni-italiani.csv` — ISTAT municipal cadastral codes

Output:
- enriched street dataset written to `input/STRAD_ITA_20251010.csv`

### 2. Start the LLM inference server

This project uses **vLLM** to serve a local OpenAI-compatible API on port `8000`.

```bash
./serve_llm.sh --gpus "0,1,2,3"
```

Notes:
- Adjust the GPU list according to your hardware
- Override the model with `--model <hf-model-id>` if needed (default: `meta-llama/Llama-3.1-70B-Instruct`)
- The server must be running before starting the classification step

### 3. Classify street dedications

This step:
- parses odonyms
- strips DUG prefixes
- applies fallback logic for unknown street types
- classifies dedications via `dedication_classifier.py` (which calls the local vLLM server)
- builds a normalized SQLite database

```bash
python classify_street_dedications.py
```

Outputs (written to the `output/` folder):
- `output/llama-3.1-70b-instruct_classifications.csv` — classified dedication labels
- `output/llama-3.1-70b-instruct_streets.sqlite` — SQLite database containing streets, municipalities, provinces, and entities

> Output filenames are derived from the model name, so they change if you serve a different model.

### 4. Generate thematic maps and reports

This step runs analytical queries on the database and produces thematic outputs focused on gender distribution.

```bash
python analyze_gender_distribution.py
```

By default it reads `output/llama-3.1-70b-instruct_streets.sqlite`
(override with `--db <path>`).

Outputs include:
- street-level and aggregated CSV tables (in `output/reports/`)
- static plots (PDF) and interactive choropleth maps (HTML) (in `output/maps/`)
- per-municipality downloadable CSV extracts
- evaluation reports and diagnostics when reference data are available

---

## 🗂️ Data Sources

- **Lombardy Regional Street Registry**
  https://www.anncsu.gov.it/it/consultazione-dellarchivio/open-data/Accedi-ai-servizi-di-dowload-massivo-in-Open-data/

- **Municipal cadastral codes (ISTAT)**
  https://www.istat.it/storage/codici-unita-amministrative/Elenco-comuni-italiani.xlsx

- **Generic Urban Street Name Registry (DUG)**
  https://registry.geodati.gov.it/dug

- **Administrative boundaries for statistical purposes (ISTAT, 1 January 2025)**
  https://www.istat.it/storage/cartografia/confini_amministrativi/generalizzati/2025/Limiti01012025_g.zip

---

## 📖 Citation

If you use this work, please cite the related paper:

```bibtex
@article{gatti2026revealing,
  title   = {Revealing gendered patterns in Italy's street names with large language models},
  author  = {Gatti, Mattia and Gallo, Ignazio and Muti, Giuseppe},
  journal = {Journal of Computational Social Science},
  volume  = {9},
  number  = {3},
  pages   = {54},
  year    = {2026},
  publisher = {Springer},
  doi     = {10.1007/s42001-026-00486-z},
  url     = {https://doi.org/10.1007/s42001-026-00486-z}
}
```
