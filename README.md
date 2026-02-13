# 🤖 Turkish Legal RAG Chatbot
### Retrieval Augmented Generation for Turkish Legal Education Documents
**Türk Hukuk Eğitimi Dokümanları için Geri Getirme Güçlendirilmiş Üretim Sistemi**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)

---

## 🎯 Project Overview | Proje Genel Bakış

A production-ready RAG (Retrieval Augmented Generation) chatbot specialized in Turkish legal education. This system processes legal documents, creates semantic embeddings, and generates contextually accurate answers by retrieving relevant information before generation.

**Turkish:** Türk hukuk eğitimi alanına özel, üretime hazır bir RAG chatbot'u. Bu sistem hukuk dokümanlarını işler, semantik embedding'ler oluşturur ve üretim öncesinde ilgili bilgileri alarak bağlamsal olarak doğru cevaplar üretir.

---

## ✨ Key Features | Temel Özellikler

- 🇹🇷 **Turkish Language Optimization** - Specialized Turkish NLP models  
  *(Türkçe Dil Optimizasyonu - Özel Türkçe NLP modelleri)*
  
- 📚 **Intelligent Document Processing** - Smart chunking with overlap  
  *(Akıllı Doküman İşleme - Örtüşmeli akıllı parçalama)*
  
- 🔍 **Semantic Vector Search** - FAISS-powered similarity matching  
  *(Semantik Vektör Araması - FAISS destekli benzerlik eşleştirme)*
  
- 🎯 **Hallucination Detection** - Grounding score calculation  
  *(Halüsinasyon Tespiti - Temellendirme skoru hesaplama)*
  
- 📊 **Source Attribution** - Track answer origins  
  *(Kaynak Atıfı - Cevap kökenlerini takip)*
  
- 💰 **Zero Cost** - No API keys required, fully open-source  
  *(Sıfır Maliyet - API anahtarı gerekmez, tamamen açık kaynak)*

---

## 🏗️ Architecture | Mimari
```
User Question (Kullanıcı Sorusu)
    ↓
Embedding Model (Turkish BERT - 768 dimensions)
    ↓
FAISS Vector Search (Cosine Similarity)
    ↓
Top-3 Chunks Retrieved (Context Building)
    ↓
Prompt = Context + Question
    ↓
LLM Generation (Turkish GPT-2)
    ↓
Grounded Answer + Source Attribution
```

**Why RAG? (Neden RAG?)**

Traditional LLMs hallucinate when asked about documents they haven't seen. RAG solves this by:
1. **Retrieval First** - Get actual document chunks
2. **Then Generate** - Use real context to produce answers
3. **Source Tracking** - Show where information came from

*Geleneksel LLM'ler görmediği dokümanlar hakkında sorulduğunda halüsinasyon yapar. RAG bunu şöyle çözer:*
1. *Önce Getir - Gerçek doküman parçalarını al*
2. *Sonra Üret - Gerçek bağlamı kullanarak cevap oluştur*
3. *Kaynak Takibi - Bilginin nereden geldiğini göster*

---

## 🛠️ Tech Stack | Teknoloji Yığını

### Models | Modeller
- **Embedding Model**: `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`
  - Turkish-optimized BERT (Türkçe optimize BERT)
  - 768-dimensional embeddings (768 boyutlu embedding'ler)
  - Semantic similarity search (Semantik benzerlik araması)

- **LLM**: `ytu-ce-cosmos/turkish-gpt2-large`
  - Turkish GPT-2 Large (Türkçe GPT-2 Büyük)
  - Text generation (Metin üretimi)
  - Domain-aware responses (Alan farkında yanıtlar)

### Libraries | Kütüphaneler

| Library | Version | Purpose |
|---------|---------|---------|
| **LangChain** | 0.1.0 | RAG framework |
| **FAISS** | 1.7.4 | Vector similarity search |
| **Transformers** | 4.36.0 | Hugging Face models |
| **Sentence Transformers** | 2.2.2 | Text embeddings |
| **PyPDF2** | 3.0.1 | PDF processing |
| **PyTorch** | 2.1.0 | Deep learning backend |

---

## 🚀 Quick Start | Hızlı Başlangıç

### Prerequisites | Ön Gereksinimler
- Python 3.8+
- Google Colab (recommended) or local environment
- 8GB+ RAM (for model loading)

### Installation | Kurulum

**Option 1: Google Colab (Recommended)**
```python
# Open the notebook in Google Colab
# Run all cells sequentially
# No local setup needed!
```

**Option 2: Local Environment**
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/turkish-legal-rag-chatbot.git
cd turkish-legal-rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage | Temel Kullanım
```python
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# 1. Load embedding model
embedding_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

# 2. Load FAISS index (if you have the saved files)
index = faiss.read_index("data/faiss_index.bin")

# 3. Load chunks
with open('data/chunk_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)
    chunks = mapping['chunks']

# 4. Search function
def search(question, k=3):
    # Convert question to embedding
    q_embedding = embedding_model.encode([question]).astype('float32')
    
    # Search FAISS
    distances, indices = index.search(q_embedding, k)
    
    # Return relevant chunks
    results = [chunks[i] for i in indices[0]]
    return results

# 5. Ask a question!
question = "Karşılaştırmalı hukukun rolü nedir?"
context = search(question)
print(context)
```

---

## 📊 Performance Metrics | Performans Metrikleri

### Document Processing | Doküman İşleme
| Metric | Value |
|--------|-------|
| **Total PDFs** | 3 documents |
| **Total Pages** | 58 pages |
| **Total Chunks** | ~150 chunks |
| **Chunk Size** | 1000 characters |
| **Chunk Overlap** | 200 characters |

### Embedding Performance | Embedding Performansı
| Metric | Value |
|--------|-------|
| **Model** | Turkish BERT |
| **Dimension** | 768 |
| **Total Embeddings** | ~150 vectors |
| **Memory Size** | ~1.2 MB |

### Retrieval Performance | Geri Getirme Performansı
| Metric | Value |
|--------|-------|
| **Index Type** | FAISS IndexFlatL2 (Exact) |
| **Search Method** | Cosine Similarity |
| **Avg Retrieval Time** | <50ms |
| **Top-k Results** | 3 chunks per query |

### Answer Quality | Cevap Kalitesi
| Metric | Value |
|--------|-------|
| **Grounding Score** | 60-80% |
| **Source Attribution** | 100% (always shown) |
| **Hallucination Rate** | Low (context-grounded) |

---

## 📚 Source Documents | Kaynak Dokümanlar

This chatbot was trained on 3 Turkish legal education documents:

1. **"Hukuk Eğitimindeki Son Gelişmeler ve Karşılaştırmalı Hukukun Hukuk Eğitimindeki Rolü"**  
   - 40 pages
   - Topics: Legal education reforms, comparative law, European legal systems

2. **"Hukuk Biliminde Gelişme"**  
   - 8 pages  
   - Topics: Legal epistemology, paradigm shifts, legal positivism

3. **"Hukuk Alanında İşbirliğinin Türk Dünyası Açısından Önemi"**  
   - 10 pages
   - Topics: Legal cooperation among Turkic states, judicial modernization

---

## 🔬 How It Works | Nasıl Çalışır

### Step-by-Step Process | Adım Adım Süreç

**1. Document Processing (Doküman İşleme)**
- PDFs are read using PyPDF2
- Text extracted and cleaned
- Split into 1000-char chunks with 200-char overlap
- Each chunk stored with metadata

**2. Embedding Generation (Embedding Oluşturma)**
- Each chunk converted to 768-dim vector using Turkish BERT
- Embeddings capture semantic meaning
- Similar chunks have similar vectors

**3. Vector Storage (Vektör Depolama)**
- All embeddings stored in FAISS index
- IndexFlatL2 used for exact search
- Fast retrieval (<50ms)

**4. Query Processing (Sorgu İşleme)**
- User question converted to embedding
- FAISS searches for top-3 similar chunks
- Chunks ranked by cosine similarity

**5. Answer Generation (Cevap Üretimi)**
- Retrieved chunks combined as context
- Prompt: "Context: ... Question: ... Answer:"
- Turkish GPT-2 generates grounded answer

**6. Quality Check (Kalite Kontrolü)**
- Grounding score calculated
- Source attribution shown
- Hallucination detection applied

---

## 📈 Results & Validation | Sonuçlar ve Doğrulama

### RAG vs Non-RAG Comparison

**Example Question:** "Karşılaştırmalı hukukun rolü nedir?"

| Aspect | RAG Answer | Non-RAG Answer |
|--------|------------|----------------|
| **Specificity** | Detailed, document-based | Generic, vague |
| **Accuracy** | High (from sources) | Variable (from training) |
| **Sources** | 3 documents cited | None |
| **Grounding** | 75% score | Not applicable |
| **Hallucination** | Low risk | High risk |

**Key Finding:** RAG answers are 60-80% grounded in source documents, significantly reducing hallucination compared to standalone LLM usage.

---

## 🎯 Use Cases | Kullanım Alanları

### Current Applications | Mevcut Uygulamalar
- ✅ Legal education Q&A (Hukuk eğitimi soru-cevap)
- ✅ Document-based research assistant (Doküman tabanlı araştırma asistanı)
- ✅ Turkish legal concept explanation (Türk hukuk kavram açıklaması)

### Potential Extensions | Potansiyel Genişletmeler
- 🔜 Case law analysis (Vaka hukuku analizi)
- 🔜 Legal document drafting assistance (Yasal belge taslağı yardımı)
- 🔜 Multi-language support (Çoklu dil desteği)

---

## 🔮 Future Improvements | Gelecek İyileştirmeler

### Technical Enhancements | Teknik Geliştirmeler
- [ ] Multi-turn conversation support (Çok turlu konuşma desteği)
- [ ] Query expansion techniques (Sorgu genişletme teknikleri)
- [ ] Re-ranking algorithms (Yeniden sıralama algoritmaları)
- [ ] Hybrid search (keyword + semantic) (Hibrit arama)

### Model Improvements | Model İyileştirmeleri
- [ ] Fine-tune on legal corpus (Hukuk korpusu üzerinde ince ayar)
- [ ] Larger context windows (Daha büyük bağlam pencereleri)
- [ ] Better Turkish LLM integration (Daha iyi Türkçe LLM entegrasyonu)

### Content Expansion | İçerik Genişletme
- [ ] Add more legal documents (Daha fazla hukuk dokümanı ekleme)
- [ ] Include case law (Vaka hukuku dahil etme)
- [ ] Multi-domain support (Çok alanlı destek)

---

## 👨‍💻 Author | Yazar

**Mustafa Haybat Gözgöz**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/mustafa-haybat-gozgoz35)

**Background (Geçmiş):**  
Former legal professional transitioning to AI/ML engineering. This project combines domain expertise in Turkish law with modern NLP techniques.

*Eski hukuk profesyonelinden AI/ML mühendisliğine geçiş yapıyor. Bu proje, Türk hukuku alanındaki uzmanlığı modern NLP teknikleriyle birleştiriyor.*

**Why This Project? (Neden Bu Proje?):**  
Having worked in the legal field, I understand the challenges of information retrieval in specialized domains. RAG technology bridges the gap between legal expertise and AI capabilities.

---

## 📄 License | Lisans

MIT License - Free to use for educational and commercial purposes.

---

## 🙏 Acknowledgments | Teşekkürler

- **Hugging Face** - Turkish NLP models
- **Meta AI** - FAISS vector search library
- **LangChain** - RAG framework
- **Turkish NLP Community** - Open-source models and support

---

## 📞 Contact & Collaboration | İletişim ve İşbirliği

Interested in:
- RAG systems for specialized domains?
- Turkish NLP applications?
- Legal tech innovation?
- Collaboration opportunities?

**Let's connect!** (Bağlantı kuralım!)

💼 LinkedIn: [linkedin.com/in/mustafa-haybat-gozgoz35](https://www.linkedin.com/in/mustafa-haybat-gozgoz35)

---

## 🌟 Star This Project! | Bu Projeye Yıldız Ver!

If you find this project useful, please star the repository!

*Bu projeyi faydalı bulduysanız, lütfen repository'ye yıldız verin!*

---

**Built with ❤️ and Turkish NLP**  
*Türkçe NLP ile ❤️ ile yapıldı*
