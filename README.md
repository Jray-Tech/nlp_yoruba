# 🧠 English–Yoruba Language Classification & Translation

This project investigates **machine learning and deep learning models** for English–Yoruba language processing. From traditional Naive Bayes classifiers to modern Transformer-based translation, it showcases a practical pipeline for low-resource language applications.

---

## 📌 Project Overview

This notebook demonstrates how to:

- 🔤 **Classify** text as English or Yoruba using **Naive Bayes**.
- 🧠 **Cluster** semantically similar sentences with **K-Means on sentence embeddings**.
- 🔁 Build a **sequence-to-sequence (seq2seq)** translation model using **RNN/LSTM**.
- 🌍 Fine-tune a **Transformer (e.g., mT5)** model for **neural machine translation**.
- 🧪 Evaluate results with industry-standard metrics: **Accuracy**, **F1-score**, **BLEU**, **METEOR**, and **Silhouette Score**.

---

## 🛠️ Technologies Used

- **Python**, **Scikit-learn**, **TensorFlow/Keras**, **Transformers (HuggingFace)**
- **NLTK**, **Pandas**, **Seaborn**, **Matplotlib**
- **sacreBLEU**, **rouge_score**, **sentencepiece**
- Pretrained GloVe embeddings for sentence vectorization

---

## 📊 Dataset

- File: `train.tsv`
- Structure: Parallel English–Yoruba sentence pairs.
- Tasks include tokenization, length analysis, and vocabulary inspection.

---

## 🧪 Evaluation Metrics

| Task                          | Evaluation Metric  |
| ----------------------------- | ------------------ |
| Classification                | Accuracy, F1-Score |
| Clustering                    | Silhouette Score   |
| Translation (RNN/Transformer) | BLEU, METEOR       |

---

## 📁 Project Structure

```
├── glove.6B.100d.txt         # Pretrained word vectors (downloaded)
├── train.tsv                 # Parallel corpus for training
├── AppliedAI.ipynb           # Main Jupyter Notebook
└── README.md                 # Project summary (this file)
```

---

## 🚀 How to Run

1. Clone the repo and install dependencies:

   ```bash
   pip install pandas scikit-learn matplotlib seaborn nltk tensorflow transformers datasets sacrebleu rouge_score
   ```

2. Launch the notebook:

   ```bash
   jupyter notebook AppliedAI.ipynb
   ```

3. Follow the cells step-by-step to preprocess data, train models, and evaluate performance.

---

## 📈 Highlights

- Clean preprocessing pipeline for multilingual data
- Demonstrates **zero-shot** translation potential using multilingual models
- Applies both **symbolic ML** and **deep learning** to a real-world, low-resource language task

---

## 🧠 Author Notes

This work was part of a broader research initiative to support **low-resource language translation** and improve **NLP inclusivity** for African languages.
