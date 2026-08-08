# HUMA PLAG - Plagiarism Detection & AI Humanization

A Python-based plagiarism detection and AI-text humanization system powered by a **fine-tuned Hugging Face Longformer** model, with **Gemini API** integration for rewriting flagged text in a more human style. The project ships with a CLI (`main.py`), a Streamlit web UI (`app.py`), and a FastAPI service (`api_integration.py`), all sharing the same core logic in `src/`.

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [How Detection Works](#-how-detection-works)
- [Configuration](#-configuration)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

- **Fine-Tuned Longformer Model** - built on `jpwahle/longformer-base-plagiarism-detection`, fine-tuned to ~92.5% accuracy on a custom binary dataset (falls back to the base pretrained model, ~81% accuracy, if no fine-tuned weights are found)
- **Grounded Predictions** - verdicts are checked against a reference corpus (`reference_corpus.py`) so "Plagiarized" calls are backed by an actual source document, reducing confident-but-wrong calls on original text
- **AI Humanization** - rewrites flagged/plagiarized text using the Gemini API (`gemini-2.5-flash`) to produce more natural, human-sounding output
- **Full Pipeline Mode** - detect plagiarism and humanize the result in a single pass
- **Multiple Interfaces**
  - 🖥️ CLI menu (`main.py`)
  - 🌐 Streamlit web app (`app.py`)
  - 🔌 FastAPI REST service (`api_integration.py`)
- **Text Comparison** - compare two texts directly for similarity
- **Synthetic Dataset Generation** - built-in tools to generate balanced original/plagiarized training data (`data_preparation.py`, `dataset_creator.py`)
- **Text Preprocessing Utilities** - stopword removal, normalization, and keyword extraction (`preprocessing.py`)
- **Model Performance & Stats** - view accuracy, F1, AUC, and session statistics from the CLI

---

## 📁 Project Structure

```
plagiarism_project/
├── data/
│   ├── plagiarism_dataset_binary.csv
│   ├── plagiarism_dataset_large.json
│   └── reference_corpus.txt
├── models/                     # Fine-tuned model checkpoints
├── src/
│   ├── api_integration.py      # FastAPI service
│   ├── data_preparation.py     # Synthetic dataset generator (v1)
│   ├── dataset_creator.py      # Synthetic dataset generator (v2, binary)
│   ├── fine_tuner.py           # Fine-tuning logic for the Longformer model
│   ├── huggingface_detector.py # Core detector + grounded prediction logic
│   ├── preprocessing.py        # Text cleaning & keyword extraction
│   ├── reference_corpus.py     # Reference corpus loader for grounding
│   ├── text_humanizer.py       # Gemini-based humanization
│   └── train.py                # Training entry point
├── training_output/
├── .env.example
├── app.py                      # Streamlit UI
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) CUDA-capable GPU for faster fine-tuning
- A Gemini API key for humanization features

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/plagiarism_project.git
cd plagiarism_project
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
```
```
GEMINI_API_KEY=your_api_key_here
```

---

## 🖥️ Usage

### Run the CLI
```bash
python main.py
```
This launches an interactive menu:
```
1.  🔍 Check Text for Plagiarism
2.  📊 Compare Two Texts
3.  ✍️  Humanize Text with Gemini
4.  🔄 Full Pipeline (Detect + Humanize)
5.  📈 Show Model Performance
6.  📊 Show Statistics
7.  🔄 Reload Model
0.  🚪 Exit
```

### Run the Streamlit web app
```bash
streamlit run app.py
```

### Run the FastAPI service
```bash
uvicorn api_integration:app --reload
```
Available endpoints include `/detect`, `/compare`, `/humanize`, `/full-pipeline`, and `/batch`.

---

## 🏋️ Model Training

1. Generate a dataset:
   ```bash
   python src/dataset_creator.py
   ```
   This creates a balanced binary dataset (`original` vs `plagiarized`) at `data/plagiarism_dataset_binary.csv`.

2. Fine-tune the model:
   ```bash
   python src/train.py
   ```
   Training runs for 5 epochs and reports accuracy, F1, and AUC after each epoch. The best checkpoint is saved to `./models/fine_tuned_model`.

**Example results from a training run:**

| Epoch | Accuracy | F1 Score | AUC |
|-------|----------|----------|--------|
| 1 | 88.50% | 0.8835 | 0.9124 |
| 2 | 90.00% | 0.8990 | 0.9187 |
| 3 | 90.50% | 0.9041 | 0.9303 |
| 5 | 91.00% | 0.9093 | 0.9318 |

---

## 🔍 How Detection Works

1. Raw input text is fed directly to the fine-tuned Longformer model (matching how it was trained - no preprocessing is applied before inference).
2. The model outputs a plagiarism probability score.
3. The prediction is **grounded** against `reference_corpus.py`: a "Plagiarized" verdict only counts if there's supporting overlap with an actual reference document, which helps prevent false positives on original, casual, or first-person text.
4. If plagiarism is detected, the text can optionally be passed to the Gemini-powered humanizer to generate a rewritten, more natural version.

> ⚠️ **Note:** The reference corpus ships with a tiny 2-sentence fallback for demo purposes. For meaningful grounding, add real source documents to `data/reference_corpus.txt` (one document per line).

---

## ⚙️ Configuration

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | API key for Gemini-based text humanization |

Model and corpus paths (e.g. `./models/fine_tuned_model`, `data/reference_corpus.txt`) can be adjusted directly in `src/huggingface_detector.py` and `src/reference_corpus.py`.

---

## 📦 Dependencies

Key libraries used in this project:
- `transformers` / `torch` - Longformer model loading & fine-tuning
- `google-generativeai` - Gemini API client
- `pandas`, `scikit-learn` - dataset handling & train/test splitting
- `fastapi`, `uvicorn` - REST API service
- `streamlit` - web UI
- `python-dotenv` - environment variable management

See `requirements.txt` for the full list and pinned versions.

---

## 🤝 Only Contributer

Jawaria Tariq (AI Engineer)

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or suggestions, feel free to open an issue on GitHub.
