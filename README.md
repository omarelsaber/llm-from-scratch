# 🧠 LLM from Scratch

This repository documents my journey of building a **Large Language Model (LLM) from scratch**, inspired by *Sebastian Raschka's book: "Build a Large Language Model (From Scratch)"*.  

The project includes both **theory (documentation)** and **hands-on implementation (code + experiments)**.  
Each chapter is implemented step by step with Jupyter Notebooks, reusable Python modules, and unit tests.

---

## 📂 Project Structure
llm-from-scratch/
│
├── data/ # Text datasets (e.g., The Verdict by Edith Wharton)
├── notebooks/ # Jupyter notebooks for each chapter
│ └── 02-tokenization-and-embeddings.ipynb
├── src/ # Core Python code (tokenizer, dataset, embeddings, etc.)
│ ├── tokenizer.py
│ ├── dataset.py
│ └── embeddings.py
├── tests/ # Unit tests for reproducibility
│ └── test_tokenizer.py
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── LICENSE

---

## 🚀 Progress
- [x] **Chapter 2**: Tokenization & Embeddings
  - Load and preprocess raw text
  - Implement regex-based tokenization
  - Build BPE tokenizer (using `tiktoken`)
  - Create a sliding-window dataset with PyTorch
  - Implement token embeddings + positional embeddings

✅ Notebook: [`02-tokenization-and-embeddings.ipynb`](notebooks/02-tokenization-and-embeddings.ipynb)

---

## 🛠️ Tech Stack
- Python 3.11+
- PyTorch
- tiktoken (OpenAI’s BPE tokenizer)
- Jupyter Notebooks
- pytest (for testing)

---

## 📌 Next Steps
- [ ] Chapter 3: Self-Attention and the Transformer architecture
- [ ] Build a minimal GPT model
- [ ] Training loop & optimization
- [ ] Scaling up to larger models

---

## 📖 References
- *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning, 2024)  
- OpenAI’s [tiktoken library](https://github.com/openai/tiktoken)  
- PyTorch documentation  

---

## 🤝 Contribution
This repo is mainly for **learning and experimentation**.  
But feel free to open issues or PRs if you have suggestions for improvements!
