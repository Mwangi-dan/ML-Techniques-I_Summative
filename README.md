# Medical Chatbot Using T5 Transformer

## Overview

This project implements a **medical chatbot** using a fine-tuned **T5 transformer model**. The chatbot takes in a user’s **symptoms** and provides a **conversational diagnosis**. Unlike traditional classifiers that provide short labels, this chatbot generates **full, natural language responses**.

The chatbot is deployed in a **Streamlit-based web interface**, allowing users to enter symptoms and receive AI-generated medical insights.

## Features

- **Fine-tuned T5 transformer model** for generating responses.
- **Conversational-style answers** instead of one-worded diagnoses.
- **Streamlit web interface** for user interaction.
- **BLEU & ROUGE evaluation metrics** to assess chatbot accuracy.

---

## Installation

To run the chatbot locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/mwangi-dan/medical-chatbot.git
cd medical-chatbot
```

### 2. Install Dependencies

Ensure you have **Python 3.7+** installed, then run:

```bash
pip install -r requirements.txt
```

This installs:

- `torch`
- `transformers`
- `streamlit`
- `rouge-score`
- `nltk`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The web interface will be accessible at `http://localhost:8501/`.

---

## Model Training & Fine-Tuning

The chatbot model is trained on a **medical symptoms dataset** with conversational response templates. The training process includes:

1. **Data Preprocessing**
   - Cleaning the text
   - Formatting responses conversationally
2. **Fine-Tuning the T5 Model**
   - Tokenizing input symptoms and responses
   - Training using PyTorch & Hugging Face Transformers
3. **Evaluation**
   - **BLEU Score** (for text similarity)
   - **ROUGE Score** (for recall-based evaluation)

To train the model from scratch, run:

```bash
python train.py
```

---

## Using the Chatbot

### 1. Running the Chatbot in Streamlit

Launch the chatbot UI with:

```bash
streamlit run app.py
```

### 2. Getting a Diagnosis

- Enter symptoms in the text box.
- Click **"Get Diagnosis"**.
- The chatbot generates a response based on learned medical conditions.

Example:

```
User: I have a fever and body aches.
Chatbot: It seems like you might be experiencing flu symptoms. I recommend getting a medical opinion to confirm.
```

---

## Evaluation Metrics

To measure chatbot performance, we use:

- **BLEU Score** (Measures similarity between predicted and actual text.)
- **ROUGE Score** (Measures recall overlap in generated text.)

Run evaluation with:

```bash
python evaluate.py
```

---

## Future Improvements

- Expand the dataset to include **more symptoms and conditions**.
- Improve response diversity using **dialogue-based fine-tuning**.
- Deploy the chatbot as a **REST API** for integration with medical platforms.

---


## Author

Developed by **Daniel Ndungu** as part of an AI-powered healthcare assistant project.

---

## Acknowledgments

- **Hugging Face** for the `T5` transformer model.
- **Streamlit** for the web-based interface.
- **PyTorch** for efficient model training.


