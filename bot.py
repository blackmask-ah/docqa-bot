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

from duckduckgo_search import DDGS  # 👈 for !ask web search
from googlesearch import search
from bs4 import BeautifulSoup

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
indexed_files = set()  # NEW: Track already indexed files

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

from duckduckgo_search import DDGS
from newspaper import Article
import nltk
nltk.download('punkt')  # Needed for newspaper

# --- Existing summarize_long_text function ---
# -*- coding: utf-8 -*-
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
    if len(combined.split()) <= 20:
        return "⚠️ Summary was too generic. Try uploading a clearer file."

    # Try to extract example sentence
    example_sentence = next(
        (s for s in sent_tokenize(text) if any(x in s.lower() for x in ["for example", "e.g.", "such as"])),
        "(No clear example found in the text.)"
    )

    # Remove emojis or replace with ASCII equivalents
    formatted = f"""Formatted Summary:

Intro:  
{summaries[0] if summaries else combined}

Example:  
{example_sentence}

Key Points:  
• {"\n• ".join(summaries[1:]) if len(summaries) > 1 else "Not enough data to split."}
"""
    return formatted.strip()



# --- Get topic from file ---
def extract_topic(text):
    return text.split('.')[0]  # crude method: first sentence


# --- Search web and summarize articles ---
def fetch_web_summaries(topic, max_articles=2):
    print(f"🔍 Searching web for: {topic}")
    results = ddg(topic, max_results=max_articles)
    summaries = []
    for res in results:
        try:
            url = res['href']
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            summaries.append(f"🔗 *{article.title}*\n{article.summary}")
        except Exception as e:
            continue
    return summaries


# --- Combined Function ---
def summarize_text_and_web(text):
    local_summary = summarize_long_text(text)
    topic = extract_topic(text)
    web_summaries = fetch_web_summaries(topic)

    formatted_web = "\n\n".join(web_summaries) if web_summaries else "🌐 No reliable web content found."

    final_output = f"""{local_summary}

*🌍 Web Insights on "{topic}":*  
{formatted_web}
"""
    return final_output.strip()


def reasoning_summary(text, doc_type):
    if doc_type in ["invoice", "resume"]:
        return f"📌 Detected: {doc_type}. Skipping summarization for structured content."
    return summarize_text(text)

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
# WEB SEARCH (!ask command)
# -----------------------------------------------------------------------------
def web_search(query, max_results=3):
    results = []
    for url in search(query, num_results=max_results, lang="en"):
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            desc = ""
            tag = soup.find("meta", attrs={"name": "description"})
            if tag and tag.get("content"):
                desc = tag["content"].strip()
            results.append(f"*{title}*\n{url}\n> {desc}")
        except Exception:
            results.append(f"*{url}*\n{url}\n> (No preview available)")
    return results or ["🤷 No English results found."]

# -----------------------------------------------------------------------------
# FILE LOGIC HANDLER
# -----------------------------------------------------------------------------
def process_file_logic(file_info, say):
    try:
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        file_path, temp_dir = download_file(file_info, headers)
        filename = file_info['name']

        if filename in indexed_files:
            say(f"ℹ️ File `{filename}` has already been indexed.")
            return

        say(f"🔍 Thinking... analyzing `{filename}`...")
        text = extract_text(file_path)
        if not text or "Unsupported" in text:
            say("❌ File is either unsupported or empty.")
            return

        text = clean_text(text)
        doc_type = detect_type(text)
        say(f"🧠 Detected document type: *{doc_type}*")

        result = reasoning_summary(text, doc_type)
        say(f"📄 *Summary of* `{filename}`:\n{result}")

        embedding = embedding_model.encode([text])[0]
        index.add(np.array([embedding]).astype('float32'))
        document_store.append((text, filename))
        indexed_files.add(filename)

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
    user_msg = event.get("text", "").lower().strip()

    if user_msg in ["hi", "hello", "hey", "help", "!help"]:
        say("""
👋 *Welcome to the BlackxMask AI Bot!*
‣ Upload `.pdf`, `.docx`, or `.txt` files  
‣ Ask questions like: `What is the summary?`  
‣ Use `!ask <question>` for web search (English only)
💡 *Example:* `!ask what is zero trust architecture?`
        """)
        return

    if user_msg.startswith("!ask"):
        query = user_msg.replace("!ask", "", 1).strip()
        if not query:
            say("❗ Please ask something like `!ask what is AI?`")
        else:
            say(f"🔍 Searching the web for: *{query}*")
            try:
                results = web_search(query)
                say("🌐 Top results:\n" + "\n\n".join(results))
            except Exception as e:
                say(f"⚠️ Web search failed: {e}")
        return

    if user_msg.endswith("?"):
        if len(document_store) == 0:
            say("📂 No documents uploaded yet. Upload a PDF, DOCX, or TXT file first.")
            return

        embedding = embedding_model.encode([user_msg])[0]
        D, I = index.search(np.array([embedding]).astype('float32'), k=5)

        seen_sources = set()
        responses = []
        for idx in I[0]:
            if idx == -1:
                continue
            context, source = document_store[idx]
            if source in seen_sources:
                continue
            seen_sources.add(source)
            snippet = context[:500].replace("\n", " ") + "..."
            responses.append(f"*From `{source}`*:\n> {snippet}")

        if not responses:
            say("🤷 I couldn't find anything relevant.")
        else:
            say("🔎 Here's what I found:\n" + "\n\n".join(responses))
        return

    if subtype == "file_share":
        try:
            file_id = event["files"][0]["id"]
            file_info = client.files_info(file=file_id)["file"]
            say("📅 File received. Let me think about it...")
            threading.Thread(target=process_file_logic, args=(file_info, say)).start()
        except Exception as ex:
            logger.error(f"Error: {ex}")
            say("⚠️ Couldn't process the uploaded file.")
        return

    say("""
🤖 *Need help using the bot?*
‣ Upload `.pdf`, `.docx`, or `.txt` file  
‣ Ask a question after uploading: `What is the summary?`  
‣ Or type `!ask <your question>` for web search
Type `help` anytime to see this again.
    """)

# -----------------------------------------------------------------------------
# BOOTSTRAP BOT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
