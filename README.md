# 📚 NoteSeek — Student Doubt Search Engine

A comprehensive **Information Retrieval (IRT) system** built with Flask that implements modern search, classification, clustering, and recommendation techniques for querying study notes across multiple IR models.

---

## ✨ Key Features

### 🔍 **Information Retrieval Models**
- **Vector Space Model (VSM)** — TF-IDF + cosine similarity ranking
- **BM25** — Probabilistic ranking with term saturation and document length normalization
- **Boolean Model** — AND/OR/NOT operator support with postings list merging
- **Language Models** — Query likelihood with Jelinek-Mercer, Laplace, and Dirichlet smoothing

### 📊 **Classification & Clustering**
- **K-Nearest Neighbor (KNN)** — Document similarity-based classification
- **Naive Bayes** — Probabilistic text classification
- **Support Vector Machines (SVM)** — Linear SVM with TF-IDF features
- **K-Means Clustering** — Unsupervised document grouping
- **Hierarchical Clustering** — Agglomerative clustering with multiple linkage methods

### 💡 **Advanced Features**
- **Rocchio Algorithm** — Query expansion via relevance feedback
- **Recommendation System** — Hybrid content-based + collaborative filtering
- **Document Upload** — Extract text from PDF and PPTX files with smart topic detection
- **Evaluation Metrics** — Precision, Recall, F-Measure, MAP, NDCG, Precision@K

### 🛠️ **NLP Preprocessing**
- Tokenization, lemmatization, stemming (Porter & Snowball)
- Stopword removal with domain-specific terms
- N-gram models (unigrams, bigrams, trigrams)
- Text cleaning and token recognition

---

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run the Application**

```bash
python app.py
```

### **3. Open in Browser**

Navigate to: **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
NoteSeek/
├── data/
│   └── notes.json                    # Document collection (IRT Units 1-5)
├── engine/
│   ├── __init__.py
│   ├── indexer.py                   # Inverted index with TF-IDF & gap encoding
│   ├── retriever.py                 # VSM + BM25 + Rocchio algorithm
│   ├── boolean_model.py             # Boolean query parsing and retrieval
│   ├── language_model.py            # Language model-based ranking
│   ├── classifier.py                # KNN, Naive Bayes, SVM classification
│   ├── clusterer.py                 # K-Means & Hierarchical clustering
│   ├── evaluator.py                 # Precision, Recall, F-Measure, MAP, NDCG
│   ├── recommender.py               # Content-based + collaborative recommendations
│   ├── uploader.py                  # PDF/PPTX extraction & dynamic indexing
│   └── preprocessor.py              # Central NLP preprocessing pipeline
├── templates/
│   └── index.html                   # Web UI
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── uploads/                         # User-uploaded PDF/PPTX files
├── app.py                           # Flask application server
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 📖 How to Use

### **Main Search** (VSM / BM25)

1. Enter a search query (e.g., "information retrieval ranking")
2. Choose model: **VSM** or **BM25**
3. Set top-K results
4. Get ranked documents with snippet highlighting

### **Boolean Search**

Use operators: `AND`, `OR`, `NOT`

Example: `"information AND retrieval NOT database"`

### **Language Model Search**

Choose smoothing technique:
- **Jelinek-Mercer** — Default, balanced relevance
- **Laplace** — Simple add-1 smoothing
- **Dirichlet** — Collection-aware smoothing

### **Text Classification**

Predict which IRT unit a text belongs to using:
- KNN, Naive Bayes, SVM, or all three

### **Clustering**

Organize notes into groups:
- **K-Means** — Fast partitioning
- **Hierarchical** — Dendrogram-based clustering

### **Recommendations**

Get similar notes or personalized suggestions based on:
- Content similarity
- User interaction history
- Hybrid filtering

### **Upload Documents**

1. Upload PDF or PPTX files
2. Text is automatically extracted
3. Topics and units auto-detected
4. Sections added to live index

---

## 🔧 Configuration

### **Flask Debug Mode**

Set environment variable to enable debug mode:

```bash
# On Windows (PowerShell)
$env:FLASK_DEBUG="true"
python app.py

# On Mac/Linux
export FLASK_DEBUG=true
python app.py
```

Default: `false` (production mode)

### **BM25 Parameters** (in `engine/retriever.py`)

```python
self.bm25_k1 = 1.5    # Term frequency saturation
self.bm25_b = 0.75    # Document length normalization
```

### **Language Model Smoothing**

- **Jelinek-Mercer** (λ=0.7) — Good for balanced relevance
- **Laplace** — Simple, adds 1 to all counts
- **Dirichlet** (μ=2000) — Collection-aware smoothing

---

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main search page |
| `/search` | POST | VSM/BM25 retrieval |
| `/boolean_search` | POST | Boolean query retrieval |
| `/language_model_search` | POST | Language model ranking |
| `/classify` | POST | Text classification |
| `/cluster` | POST | Document clustering |
| `/evaluate` | POST | Evaluation metrics |
| `/recommend` | POST | Get recommendations |
| `/upload` | POST | Upload PDF/PPTX files |
| `/document/<id>` | GET | Get full document |
| `/documents` | GET | List all documents |
| `/index_stats` | GET | Index statistics |

---

## 📦 Dependencies

```
Flask           — Web framework
scikit-learn    — Classification, clustering, TF-IDF
NLTK            — Tokenization, stemming, lemmatization
NumPy / SciPy   — Numerical computing
pdfplumber      — PDF text extraction
python-pptx     — PowerPoint text extraction
Werkzeug        — Secure file handling
```

All listed in `requirements.txt`

---

## 🎓 IRT Curriculum Coverage

This project implements techniques from a 5-unit Information Retrieval course:

- **Unit 1** — IR Fundamentals & System Architecture
- **Unit 2** — Retrieval Models (Boolean, VSM, Probabilistic, Language Models) & Evaluation
- **Unit 3** — Indexing (Inverted Files, Encoding), Classification, Clustering
- **Unit 4** — Web Search & PageRank *(framework ready)*
- **Unit 5** — NLP & Preprocessing, Recommendations, Document Scoring

---

## 📚 Algorithm Details

### **Vector Space Model (VSM)**

Ranks documents by cosine similarity with query:
```
score(q, d) = cos(q_vector, d_vector) = (Q · D) / (|Q| × |D|)
```

### **BM25 (Okapi)**

Probabilistic ranking with saturation:
```
BM25(t, d, q) = IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1(1 - b + b(|d|/avgdl))]
```

### **Boolean Retrieval**

Combines postings lists using set operations:
- AND → Intersection
- OR → Union
- NOT → Complement

### **Language Model**

Query likelihood scoring with smoothing:
```
P(q|d) = Π P(term|d)^count(term,q)
```

### **K-Means Clustering**

Partitioning with centroid-based reassignment:
1. Initialize K random centroids
2. Assign documents to nearest centroid
3. Recompute centroids
4. Repeat until convergence

### **KNN Classification**

Majority vote from K nearest neighbors using cosine similarity on TF-IDF vectors

---

## 🔐 Security Features

- ✅ `secure_filename()` for file uploads
- ✅ File type validation (PDF/PPTX only)
- ✅ No hardcoded API keys or secrets
- ✅ NLTK resource errors handled gracefully
- ✅ Optional library imports with try/except
- ✅ 32 MB max upload size limit

---

## 🛠️ Troubleshooting

### **NLTK Resources Missing**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **pdfplumber Not Installed**

```bash
pip install pdfplumber
```

### **Port 5000 Already in Use**

Edit `app.py` line 290:
```python
app.run(debug=False, port=8000)  # Change to 8000
```

### **OneDrive Path Issues** (Windows)

Store project outside OneDrive or use WSL2 for better performance.

---

## 📝 File Upload Guidelines

- **Supported formats:** PDF, PPTX
- **Max file size:** 32 MB
- **Text extraction:** Automatic page/slide splitting
- **Topic detection:** Auto-detected from content
- **Unit classification:** Keyword-based detection

---

## 🎯 Example Queries

### VSM/BM25 Search
```
"TF-IDF vector space model ranking"
"boolean retrieval inverted index"
"language model smoothing"
```

### Boolean Search
```
"information AND retrieval NOT database"
"classification OR clustering"
"stemming AND lemmatization NOT vectorization"
```

### Classification
```
"How do I implement K-means clustering?"
"Explain the BM25 ranking function"
"What is precision and recall?"
```

---

## 🚀 Future Enhancements

- Web crawler for live indexing
- PageRank implementation for web search
- Sentiment analysis module
- Advanced query processing (phrases, proximity)
- Performance benchmarking suite
- User authentication and personalization
- Mobile app version

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional IR models
- Performance optimizations
- UI/UX enhancements
- Better documentation
- Test coverage

---

## 📞 Support

For issues or questions:
1. Check module docstrings (each file has comprehensive documentation)
2. Review function comments explaining algorithms
3. Examine `data/notes.json` for expected document format
4. Open an issue on GitHub

---

## 👨‍💻 Author

**Chaitanya697**

---

**Built with ❤️ for learning Information Retrieval** 📚

Made with Flask, scikit-learn, NLTK, and passion for IR!
