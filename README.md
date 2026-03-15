# AI Recipe Assistant using Gemini, FAISS, and Retrieval-Augmented Generation (RAG)

This project is an **AI-powered recipe assistant** that answers food-related questions by combining:

- **Google Gemini embeddings**
- **FAISS vector search**
- **Gemini text generation**
- **Recipe dataset retrieval**
- **RAG (Retrieval-Augmented Generation)**

The assistant retrieves the most relevant recipe entries from a dataset and uses them as references to generate a clear, non-technical response to the user.

---

## Project Overview

This project allows users to ask food-related questions such as:

- How to make broccoli rice?
- What can I cook with chicken and rice?
- Give me simple pasta ideas
- Suggest a recipe using vegetables

The workflow is:

1. Load recipe dataset
2. Generate embeddings for recipe entries
3. Store embeddings in a FAISS index
4. Convert user query into an embedding
5. Retrieve the most similar recipes
6. Send retrieved references to Gemini
7. Generate a natural-language answer

---

## Features

- Semantic recipe search using embeddings
- Fast retrieval with FAISS
- Gemini-powered answer generation
- Cached embeddings using pickle
- Interactive command-line chatbot
- Simple RAG pipeline for food and recipe queries

---

## Tech Stack

- Python
- Google Gemini API
- FAISS
- Pandas
- NumPy
- Pickle
- python-dotenv
- ChromaDB embedding interface

---

## Project Structure

```bash
.
├── app.py                    # Main chatbot / query script
├── dataset.csv               # Recipe dataset
├── saved_embeddings.pkl      # Cached embeddings
├── .env                      # Stores Gemini API key
└── README.md
