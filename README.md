# financial-sentiment-analysis
📰 Financial Sentiment Analysis in News Headlines

Team Members:
	•	K2453494 – Adharsh Vaiapuri
	•	K2501085 – Avez Mushtaq Kazi

⸻

🎯 Project Overview

Financial markets are highly sensitive to news. Even subtle wording in financial headlines can influence investor sentiment and, in turn, market behavior.

This project applies Natural Language Processing (NLP) techniques and transformer-based deep learning models to automatically classify financial news headlines into positive, neutral, or negative sentiments.

The goal is to build an efficient and accurate sentiment classification model that can assist in tasks like market trend analysis, risk assessment, and automated trading insights.

⸻

🧠 Problem Statement

Traditional sentiment models struggle with financial jargon and subtle tone variations.
This project aims to fine-tune and compare two state-of-the-art transformer models:
	•	BERT (Bidirectional Encoder Representations from Transformers)
	•	DistilBERT (a smaller, faster distilled version of BERT)

Both models are trained using the Hugging Face transformers library and evaluated on labelled financial news data.

⸻

📊 Dataset
	•	Size: ~10,000 financial news headlines
	•	Split: 70% Train | 15% Validation | 15% Test
	•	Labels:
	•	0 → Negative
	•	1 → Neutral
	•	2 → Positive

Exploratory Data Analysis (EDA):
	•	Label distribution via countplot and pie chart
	•	Word clouds for each sentiment class
	•	Sentence length distribution
	•	Visualization of sample text vs. sentiment label

⸻

⚙️ Methodology

🧩 1. Preprocessing
	•	Lowercasing
	•	Removal of special characters & extra spaces
	•	Uniform label encoding
	•	Tokenization with AutoTokenizer

🔍 2. Model Fine-Tuning

BERT (bert-base-uncased)
	•	Model: BertForSequenceClassification
	•	Batch size: 4 (train), 8 (eval)
	•	Epochs: 1
	•	Optimizer: AdamW

DistilBERT
	•	Model: DistilBertForSequenceClassification
	•	Batch size: 8 (train), 16 (eval)
	•	Epochs: 3
	•	Optimizer: AdamW

Both trained using the Hugging Face Trainer API with a custom PyTorch Dataset class.


📈 Results & Analysis
Model
Training Time
Accuracy
Parameters
BERT
~40 min
96.0%
110M
DistilBERT
~15 min
96.0%
66M



Key Insights:
	•	BERT offers stronger contextual understanding and slightly better precision.
	•	DistilBERT achieves comparable accuracy with 2.5× faster training and less GPU memory usage.
	•	Both models effectively capture nuanced financial tone.

Confusion Matrix Analysis:
	•	BERT performs better at separating neutral vs. positive sentiment.
	•	DistilBERT shows minor confusion between neutral and negative headlines.


🧾 Implementation Notes
	•	Frameworks: PyTorch, Hugging Face Transformers
	•	Visualization: Matplotlib, Seaborn, WordCloud
	•	GPU Tested: NVIDIA T4 / A100
	•	Modular and documented notebook with debugging checkpoints

⸻

🧩 Key Takeaways
	•	BERT: Best for high-stakes financial sentiment tasks where accuracy matters most.
	•	DistilBERT: Excellent balance of speed and performance — ideal for real-time or resource-limited systems.
	•	Demonstrates how transformer models outperform classical NLP in domain-specific sentiment tasks.
  
🏁 Conclusion

Both BERT and DistilBERT proved highly effective for financial sentiment analysis, achieving 96% accuracy.
While BERT excels in contextual depth, DistilBERT is a practical alternative for faster, resource-efficient deployment.

This project demonstrates how transformer-based NLP models can empower intelligent financial decision systems through text sentiment analysis.
