import streamlit as st
import pandas as pd
import numpy as np
import re, string, torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

ab = pd.read_csv("ABIEMO_2334.csv").drop('class', axis=1)
ab = ab.replace({'Emotions': 'emotions'})
bc = pd.read_csv("BAPEMO_6346.csv")
bc.rename(columns={'cleaned': 'poems'}, inplace=True)
all_poems = pd.concat([ab, bc], ignore_index=True).head(200)
Poems = all_poems["poems"]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub("<.*?>", "", text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

all_poems["poems"] = Poems.apply(clean_text)

embed_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
embed_model = AutoModel.from_pretrained("distilroberta-base").to(device)
embed_model.eval()

def get_embedding(text):
    if not text:
        return np.zeros(768)
    inputs = embed_tokenizer(text, return_tensors="pt",
                             truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        output = embed_model(**inputs).last_hidden_state
    return output[:, 0, :].cpu().numpy()[0]

all_poems["emotion_emb"] = all_poems["Emotions"].apply(get_embedding)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

def find_similar_poem(emotion):
    target = get_embedding(emotion)
    vectors = np.vstack(all_poems["emotion_emb"])
    sims = cosine_similarity([target], vectors)[0]
    idx = np.argmax(sims)
    return all_poems.iloc[idx]["poems"]

def generate_poem(emotion):
    context = find_similar_poem(emotion)
    prompt = f"Write an emotional poem expressing '{emotion}'. Use this poem as inspiration:\n{context}\nPoem:\n"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = gpt2_model.generate(inputs, max_new_tokens=60,
                                     do_sample=True, temperature=0.8, top_p=0.95)
    return gpt2_tokenizer.decode(output[0])[len(prompt):].strip()

st.title("🎭 Emotion-Based Poem Generator")
emotion = st.text_input("Enter an emotion (joy, sadness, love, fear, anger):")

if st.button("Generate Poem"):
    if emotion.strip() == "":
        st.warning("Please enter an emotion before generating!")
    else:
        poem = generate_poem(emotion)
        st.subheader("Generated Poem")
        st.write(poem)
