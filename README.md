# 🚀 DocQA Bot – Slack AI Assistant for Document Intelligence

> Built for the **AI Agent Hackathon** by Team **blackXmask**

<img width="1478" height="745" alt="p2" src="https://github.com/user-attachments/assets/d70ce23b-1906-4cc0-9830-b45bb492cfdb" />
<!-- Replace with your image path -->

---

## 🔍 What It Does

DocQA Bot is an AI-powered assistant integrated with Slack. It helps support teams instantly summarize, understand, and answer questions from complex documents — saving time and boosting response quality.

---

## ✨ Features

✅ Upload and analyze `.pdf`, `.docx`, or `.txt` files  
✅ Summarizes documents using Pegasus + chunking  
✅ Semantic search and question answering with FAISS  
✅ Web Q&A via `!ask` command (DuckDuckGo + Google)  
✅ OCR fallback for scanned PDFs  
✅ Slack-native interface with real-time interaction  
✅ 100% free and open-source (no OpenAI key needed)

---

## 🧠 Tech Stack

- **Slack Bolt (Python)** – Slack bot interface  
- **LangChain (lightweight use)** – LLM orchestration  
- **FAISS** – Vector similarity search  
- **Transformers** – Pegasus summarization model  
- **SentenceTransformers** – MiniLM for embedding  
- **OCR** – PyMuPDF + Tesseract  
- **DuckDuckGo + Google Search** – Web Q&A fallback  
- **Newspaper3k** – Web article summarization  
- **Python + NLTK + BeautifulSoup** – Utilities

---

## 🖼 Testing Screenshots

<img width="1920" height="1080" alt="Screenshot_2025-07-21_20_18_03" src="https://github.com/user-attachments/assets/eb376fac-8d22-4eb1-89e4-ccd714acadc8" />
<img width="1920" height="1080" alt="Screenshot_2025-07-21_20_17_53" src="https://github.com/user-attachments/assets/3d6d13ad-8120-4a87-b23e-04b3033a5817" />
<img width="1920" height="1080" alt="Screenshot_2025-07-21_20_17_51" src="https://github.com/user-attachments/assets/35b6d294-2375-4535-982d-5a6df2230791" />
<img width="1920" height="1080" alt="Screenshot_2025-07-21_20_17_26" src="https://github.com/user-attachments/assets/e37fa485-ece3-4001-952f-e1d451b24189" />

---

## 🚀 How to Run Locally

### 1. Clone the repo:
```bash
git clone https://github.com/your_username/docqa-bot.git
cd docqa-bot
