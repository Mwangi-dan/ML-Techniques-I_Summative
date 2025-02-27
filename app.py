import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./medical_chatbot_t5"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate chatbot responses
def chatbot_response(input_text):
    model.eval()
    input_text = "chatbot: " + input_text
    encoding = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(device)
    
    output = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit UI Setup
st.title("🩺 Medical Chatbot")
st.write("Enter your symptoms below and get a possible diagnosis.")

# User input
txt_input = st.text_area("Describe your symptoms:")

if st.button("Get Diagnosis"):
    if txt_input:
        response = chatbot_response(txt_input)
        st.subheader("Chatbot Diagnosis:")
        st.write(response)
    else:
        st.warning("Please enter symptoms before clicking Get Diagnosis.")