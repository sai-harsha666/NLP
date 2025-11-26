A deep learning project that generates poems based on a given emotion by using similarity-based poem retrieval + GPT-2 poem generation.

Project Overview:

Loads two poem datasets (ABIEMO_2334.csv, BAPEMO_6346.csv)
Cleans poem text and merges the datasets.
Extracts text embeddings using DistilRoBERTa.
Finds the most similar poem for the given emotion.
Feeds the retrieved poem into GPT-2 to generate a new poem.

How to run locally:

Install dependencies:pip install -r requirements.txt
Run the app:python poem_generate.py

Dataset:

"C:\Users\INVSS Jaswanth\OneDrive\Desktop\ABIEMO_2334.csv"

"C:\Users\INVSS Jaswanth\OneDrive\Desktop\BAPEMO_6346.csv"
