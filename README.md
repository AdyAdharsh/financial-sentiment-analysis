# Financial Sentiment Analysis in News Headlines

## Team Members
- **K2453494** – Adharsh Vaiapuri
- **K2501085** – Avez Mushtaq Kazi

---

## Project Overview

Financial markets react strongly to news headlines, which can quickly influence investor attitudes and market dynamics. Accurate analysis of sentiment in these headlines is critical for understanding and predicting trends in the financial sector.

This project uses advanced Natural Language Processing (NLP) and deep learning techniques to classify financial news headlines according to their expressed sentiment—positive, neutral, or negative.

**Objectives:**
- Develop a reliable, automated sentiment classification model.
- Support use cases such as market trend detection, risk analysis, and algorithmic trading.
- Compare transformer models to determine which is best suited for financial domain text.

---

## Problem Statement

Most generic sentiment analysis tools are not optimized for financial language, which often contains technical jargon and subtle tonal differences. To address this, two leading transformer models are fine-tuned and evaluated on financial data:

- **BERT (Bidirectional Encoder Representations from Transformers):** Known for its deep contextual understanding.
- **DistilBERT:** A compressed, faster variant of BERT offering good accuracy with less computational requirement.

Both models are implemented using the Hugging Face Transformers library and trained on systematically labeled financial headlines.

---

## Dataset

- **Size:** ~10,000 financial news headlines
- **Split:** 
  - 70% Training
  - 15% Validation
  - 15% Test
- **Labels:** 
  - 0: Negative
  - 1: Neutral
  - 2: Positive

**Exploratory Data Analysis (EDA):**
- Counting label frequency (bar charts, pie charts)
- Generating word clouds for each class
- Reviewing sentence length patterns
- Visualizing sample headlines by sentiment

---

## Methodology

### 1. Data Preprocessing
- Convert text to lowercase.
- Remove special characters and unnecessary spaces.
- Encode sentiment labels to numerical values.
- Tokenize text using transformer-compatible tokenizers.

### 2. Model Training & Fine-Tuning

#### BERT
- Model: `BertForSequenceClassification`
- Training: Batch size 4, 1 epoch
- Evaluation: Batch size 8
- Optimizer: AdamW

#### DistilBERT
- Model: `DistilBertForSequenceClassification`
- Training: Batch size 8, 3 epochs
- Evaluation: Batch size 16
- Optimizer: AdamW

Training is performed via Hugging Face `Trainer` API, using custom PyTorch datasets to format headlines for model input.

---

## Results & Analysis

| Model      | Training Time | Accuracy | Parameters |
|------------|:-------------:|:--------:|:----------:|
| BERT       | ~40 min       | 96.0%    | 110M       |
| DistilBERT | ~15 min       | 96.0%    | 66M        |

**Model Insights:**
- BERT provides more thorough contextual analysis and excels in distinguishing subtle sentiment differences, but is slower and requires more resources.
- DistilBERT achieves the same accuracy as BERT, trains 2.5× faster, and uses less memory, making it a practical choice for real-time applications.

**Confusion Matrix Insights:**
- BERT is better at separating neutral from positive sentiment.
- DistilBERT occasionally confuses neutral with negative headlines.

---

## Skills Exhibited

- **Data Analysis & Visualization:** Mastered EDA using countplots, pie charts, word clouds, and sentence statistics.
- **NLP Preprocessing:** Skilled in cleaning and preparing sophisticated financial text data for deep learning models.
- **Model Fine-Tuning:** Proficient in training, evaluating, and comparing transformer-based models using PyTorch and Hugging Face.
- **Performance Evaluation:** Experienced in using accuracy, confusion matrices, and efficiency measurements to assess model effectiveness.
- **Deep Learning Deployment:** Familiar with deploying models for GPU acceleration (NVIDIA T4 / A100), optimizing resource usage for production scenarios.
- **Report Writing & Documentation:** Produced clear, detailed documentation and modular notebook structures with debugging checkpoints.

---

## Key Takeaways

- **BERT:** Best suited for tasks requiring deep contextual sentiment analysis, such as high-value trade or risk assessment.
- **DistilBERT:** Ideal for applications where speed and efficiency are prioritized, such as real-time analytics or systems with limited hardware.
- **Advanced Transformers:** Outperform traditional NLP techniques in handling the specialized language of financial headlines.

---

## Conclusion

Both BERT and DistilBERT offer high accuracy for financial sentiment analysis, with subtle trade-offs between speed and contextual performance. This project demonstrates the power of transformer-based NLP models to improve financial decision-making tools by accurately interpreting the sentiment of market-moving headlines.
