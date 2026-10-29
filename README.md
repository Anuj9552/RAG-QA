# RAG PDF Question Answering

एक beginner-friendly, local-first app: PDF/TXT upload करें, documents को chunks में बाँटें, Chroma में local embeddings के साथ index करें, और उनके आधार पर सवाल पूछें।

## यह कैसे काम करता है

1. **Upload और extraction:** `pypdf` searchable PDF text निकालता है; UTF-8 `.txt` files भी supported हैं।
2. **Chunking:** text word-based chunks में बाँटा जाता है। overlap chunk boundaries पर context बचाता है।
3. **Embeddings और vector DB:** Chroma का local ONNX embedding model vectors बनाता है और उन्हें `data/chroma/` में persist करता है।
4. **Retrieval:** प्रश्न के सबसे करीब passages cosine distance के आधार पर खोजे जाते हैं।
5. **Answer:** Extractive mode सबसे relevant passage दिखाता है और किसी chat model की जरूरत नहीं होती। वैकल्पिक Ollama mode retrieved passages से local LLM answer बनाता है; दोनों modes sources दिखाते हैं।

आपके documents और vector index इस project के `data/` folder में रहते हैं। पहले embedding run पर Chroma को `all-MiniLM-L6-v2` का लगभग 79 MB model download करने के लिए internet की जरूरत पड़ सकती है; उसके बाद embeddings local run होते हैं। Ollama mode पूरी तरह local है, बशर्ते Ollama install हो और चुना हुआ model मौजूद हो।

## 1. Environment setup

इस workspace में पहले से `venv` मौजूद है। Windows PowerShell में project folder खोलकर चलाएँ:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

नए clone पर virtual environment बनाएँ:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.11+ उपयोग करें। VS Code project interpreter को `.vscode/settings.json` के जरिए इस `venv` पर सेट करता है। अगर नया `venv` बनाया है, तो command palette से **Python: Select Interpreter** चलाकर `venv\Scripts\python.exe` चुनें।

## 2. App चलाएँ

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Streamlit स्थानीय browser page खोलेगा। पहला test करने के लिए **Index the sample leave policy** दबाएँ; फिर सवाल पूछें, जैसे “How many casual leave days are provided?”

## 3. अपने PDF से सवाल पूछें

1. **Choose PDF or TXT files** में एक या अधिक files चुनें।
2. आवश्यकता हो तो sidebar में chunk size और overlap बदलें।
3. **Index selected documents** दबाएँ।
4. indexed passages से सवाल पूछें। उत्तर के नीचे **Sources** खोलकर matching passages, filename, page और vector distance देखें।

Uploaded file में searchable text होना चाहिए। केवल image/scanned pages वाले PDF को पहले OCR करना होगा। 25 MB से बड़ी files स्वीकार नहीं होतीं। उसी filename को दोबारा index करने पर पुरानी copy replace होती है।

## 4. Local LLM (वैकल्पिक)

Extractive mode बिना LLM के चलता है। बेहतर natural-language answers के लिए [Ollama](https://ollama.com/) install करें, फिर PowerShell में:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

Ollama चलने पर sidebar में **Ollama local LLM** चुनें और model name `qwen2.5:3b` रखें। यदि Ollama पहले से background में चल रहा है, `ollama serve` दोबारा चलाने की जरूरत नहीं। Ollama URL और model को `OLLAMA_BASE_URL` और `OLLAMA_MODEL` environment variables से भी बदला जा सकता है।

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests PDF/text ingestion rules, chunk overlap, replacement indexing, retrieval metadata, extractive answers और Ollama failure handling जाँचते हैं। Ollama model या network call की जरूरत नहीं है।

Python syntax check:

```powershell
.\venv\Scripts\python.exe -m py_compile streamlit_app.py rag_pipeline.py tests\test_rag_pipeline.py
```

## Troubleshooting

- **`python` command नहीं मिलता:** इस workspace में global Python command उपलब्ध न भी हो सकता है। ऊपर दिए गए `.\venv\Scripts\python.exe` commands इस्तेमाल करें।
- **Ollama connection error:** Ollama app/server चलाएँ, `ollama list` से model देखें, और sidebar URL की जाँच करें।
- **Model not found:** `ollama pull qwen2.5:3b` चलाएँ या sidebar में installed model का सही नाम दें।
- **PDF में कोई text नहीं:** यह अक्सर scanned/image-only PDF होता है; OCR के बाद searchable PDF upload करें।
- **Embedding model download error:** पहली बार Chroma का ONNX model download करने के लिए internet access दें; फिर app दोबारा चलाएँ।
- **Index हटाना:** Streamlit बंद करें और `data\chroma\` folder हटाएँ। यह local vector index है; मूल PDFs इससे delete नहीं होते।

## Project structure

```text
.
├── .streamlit\config.toml
├── .vscode\settings.json
├── documents\leave_policy.txt
├── rag_pipeline.py
├── requirements.txt
├── streamlit_app.py
└── tests\test_rag_pipeline.py
```

`venv/` और `data/` को `.gitignore` में रखा गया है; GitHub पर Python environment या indexed document data commit नहीं होंगे।
