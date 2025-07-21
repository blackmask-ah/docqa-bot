# 🚀 Ultimate Slack AI Bot for Document Intelligence (Summarization + Semantic Q&A)
# -----------------------------------------------------------------------------
# ✅ Handles large PDFs, DOCX, TXT
# ✅ Summarizes with RAG-style chunking
# ✅ Uses FAISS for document Q&A with embeddings
# ✅ OCR fallback for scanned files
# ✅ Free & open-source tech only (no OpenAI key required)

# -----------------------------------------------------------------------------
# IMPORTS AND SETUP
# -----------------------------------------------------------------------------
import os
import re
import shutil
import tempfile
import threading
import logging
import requests
import concurrent.futures

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------------------------------------------------------
# AI MODELS INITIALIZATION
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "google/pegasus-xsum"  # Best free summarizer model for short-form

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)
document_store = []  # Stores (text, filename)

classifier = pipeline("zero-shot-classification", device=0 if torch.cuda.is_available() else -1)

# -----------------------------------------------------------------------------
# SLACK SETUP
# -----------------------------------------------------------------------------
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(token=SLACK_BOT_TOKEN)

# -----------------------------------------------------------------------------
# TEXT UTILITIES
# -----------------------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'(\b\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------------------------------------------------------
# SUMMARIZATION (WITH CHUNKING)
# -----------------------------------------------------------------------------
def chunk_text(text, max_tokens=800):
    sentences = sent_tokenize(text)
    chunks, current = [], []
    current_len = 0

    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))
        if current_len + token_count > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += token_count

    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize_long_text(text):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=60,
            num_beams=5,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    combined = " ".join(summaries)
    return combined if len(combined.split()) > 20 else "⚠️ Summary was too generic. Try uploading a clearer file."

def summarize_text(text):
    if len(text) < 50:
        return "Text too short to summarize."
    try:
        return summarize_long_text(text)
    except Exception as e:
        return f"⚠️ Error in summarization: {str(e)}"

def detect_type(text):
    labels = ["legal contract", "technical report", "email", "invoice", "resume", "academic paper", "generic article"]
    result = classifier(text[:1000], labels)
    return result["labels"][0]

def reasoning_summary(text, doc_type):
    if doc_type in ["invoice", "resume"]:
        return f"📌 Detected: {doc_type}. Skipping summarization for structured content."
    return summarize_text(text)

# -----------------------------------------------------------------------------
# FILE EXTRACTION / OCR
# -----------------------------------------------------------------------------
def extract_text(file_path):
    try:
        if file_path.endswith(".pdf"):
            return extract_pdf_text(file_path)
        elif file_path.endswith(".docx"):
            return "\n".join(p.text for p in Document(file_path).paragraphs)
        elif file_path.endswith(".txt"):
            return open(file_path, "r", encoding="utf-8").read()
        else:
            return "Unsupported file type."
    except Exception as e:
        return f"Error reading file: {e}"

def extract_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text if text.strip() else perform_ocr_on_pdf(file_path)
    except:
        return perform_ocr_on_pdf(file_path)

def perform_ocr_on_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    return full_text

def download_file(file_info, headers):
    url = file_info["url_private"]
    filename = file_info["name"]
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    response = requests.get(url, headers=headers)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path, temp_dir

# -----------------------------------------------------------------------------
# MAIN FILE LOGIC HANDLER
# -----------------------------------------------------------------------------
def process_file_logic(file_info, say):
    try:
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        file_path, temp_dir = download_file(file_info, headers)
        say(f"🔍 Thinking... analyzing `{file_info['name']}`...")

        text = extract_text(file_path)
        if not text or "Unsupported" in text:
            say("❌ File is either unsupported or empty.")
            return

        text = clean_text(text)
        doc_type = detect_type(text)
        say(f"🧠 Detected document type: *{doc_type}*")

        result = reasoning_summary(text, doc_type)
        say(f"📄 *Summary of* `{file_info['name']}`:\n{result}")

        embedding = embedding_model.encode([text])[0]
        index.add(np.array([embedding]).astype('float32'))
        document_store.append((text, file_info['name']))

    except Exception as e:
        say(f"⚠️ Error processing file: {e}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# -----------------------------------------------------------------------------
# MESSAGE HANDLER
# -----------------------------------------------------------------------------
@app.event("message")
def handle_message_events(event, say, logger):
    subtype = event.get("subtype", "")
    user_msg = event.get("text", "").lower()

    if any(greet in user_msg for greet in ["hi", "hello", "hey", "yo"]):
        say("👋 Hello! Upload a PDF, DOCX, or TXT file — I’ll extract, classify, and summarize it for you!")

    elif user_msg.endswith("?"):
        embedding = embedding_model.encode([user_msg])[0]
        D, I = index.search(np.array([embedding]).astype('float32'), k=3)

        if I[0][0] == -1 or len(document_store) == 0:
            say("🤷 I couldn't find anything relevant.")
            return

        responses = []
        for idx in I[0]:
            context, source = document_store[idx]
            snippet = context[:500].replace("\n", " ") + "..."
            responses.append(f"*From `{source}`*:\n> {snippet}")

        say("🔎 Here's what I found:\n" + "\n\n".join(responses))

    if subtype == "file_share":
        try:
            file_id = event["files"][0]["id"]
            file_info = client.files_info(file=file_id)["file"]
            say("📅 File received. Let me think about it...")
            threading.Thread(target=process_file_logic, args=(file_info, say)).start()
        except Exception as ex:
            logger.error(f"Error: {ex}")
            say("⚠️ Couldn't process the uploaded file.")

# -----------------------------------------------------------------------------
# BOOTSTRAP BOT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
