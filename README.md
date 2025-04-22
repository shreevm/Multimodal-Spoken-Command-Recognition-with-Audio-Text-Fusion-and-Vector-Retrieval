#  Multimodal Transformer for Spoken Command Recognition

##  Overview

This project presents a **multimodal spoken command classification system** that fuses **audio features from Wav2Vec2** and **semantic label embeddings from BERT**. The model enhances recognition accuracy and robustness by integrating both **acoustic and textual cues**, using a hybrid Transformer decoder with cross-attention.

###  What’s Special?
- Fuses **speech and semantics** (audio + text)
- Retrieves relevant **label embeddings via Pinecone vector DB**
- Improves **classification robustness** over audio-only models
- Uses **frozen pretrained encoders** for efficient training

---

##  Directory Structure


- [README.md](README.md) — Main documentation for the project  
- [Poster_presentation.pdf](Poster_presentation.pdf) — Final presentation poster  
- [spoken_command_classification.ipynb](spoken_command_classification.ipynb) — Multimodal classification training and evaluation notebook  
- [pinecone_retrieval.ipynb](pinecone_retrieval.ipynb) — Pinecone-based semantic embedding search notebook  
- [assets/demo_screenshot.png](assets/demo_screenshot.png) — Screenshot of the Gradio inference interface  
---

## Dataset

- **Name**: Google Speech Commands v0.01  
- **Source**: [`huggingface.co/datasets/speech_commands`](https://huggingface.co/datasets/speech_commands)
- **Classes**: 35 commands (e.g., yes, no, stop, left, right)
- **Preprocessing**:
  - Resampled to 16kHz
  - Normalized waveforms
  - Text labels tokenized for BERT encoder

---

##  Pipeline Steps

### Step 1: Problem Definition  
**Goal**: Improve spoken command recognition by combining both audio inputs and text label information, and retrieving semantic vectors from a Pinecone DB.

### Step 2: Model Architecture

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Audio Encoder**| Pretrained **Wav2Vec2** (frozen) extracts audio embeddings                  |
| **Text Encoder** | Pretrained **BERT-base** (frozen) generates command label embeddings        |
| **Fusion**       | Cross-attention-based **Transformer decoder**                              |
| **Retrieval**    | Uses **Pinecone vector DB** to retrieve nearest label embeddings to fuse    |

### Step 3: Training Strategy

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW` with learning rate `1e-4`
- Encoders frozen for stability and faster training
- Fusion via concatenation + attention

---

## Results

| Metric        | Value     |
|---------------|-----------|
| Train Accuracy | ~92.5%    |
| Test Accuracy  | ~91.8%    |
| F1 Score       | ~91.4%    |

- Fusion (audio + text) outperformed audio-only baseline  
- Model remained stable due to frozen encoders  
- Supports fast inference using Gradio + semantic search via Pinecone

---

## Experiments & Tasks

- `spoken_command_classification.ipynb`: Train & evaluate the multimodal Transformer.
- `pinecone_retrieval.ipynb`: Store and query command label embeddings for fusion.

---

## Setup & Installation

###  Requirements

```bash
pip install torch torchaudio transformers gradio librosa evaluate datasets pinecone-client
```

# Clone repo
``` bash
git clone https://github.com/UF-EGN6217-Spring25/project-2-shreevm
```
cd project-2-shreevm

# Run Jupyter notebooks
spoken_command_classification.ipynb
pinecone_inference.ipynb

## Summary
This project demonstrates the effectiveness of multimodal Transformer architectures in improving spoken command classification. By leveraging both acoustic patterns and semantic embeddings, the model achieves strong generalization, stability, and interpretability.


##  Demo Screenshot

Here’s a look at the Gradio interface used to test audio commands:


![Gradio Demo](assets/demo_screenshot.png)

