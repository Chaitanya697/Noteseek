"""
app.py
------
NoteSeek — Student Doubt Search Engine
Flask application server with routes for all IRT features.
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import all engine modules
from engine.indexer import Indexer
from engine.retriever import Retriever
from engine.boolean_model import BooleanModel
from engine.language_model import LanguageModel
from engine.classifier import Classifier
from engine.clusterer import Clusterer
from engine.evaluator import Evaluator
from engine.recommender import Recommender
from engine.uploader import Uploader

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'ppt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------------ #
#  INITIALIZE ALL ENGINE COMPONENTS ON STARTUP                        #
# ------------------------------------------------------------------ #

print("=" * 55)
print("  NoteSeek — IRT Search Engine")
print("  Initializing all modules...")
print("=" * 55)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'notes.json')

indexer    = Indexer(DATA_PATH)
retriever  = Retriever(indexer)
boolean    = BooleanModel(indexer)
lang_model = LanguageModel(indexer)
classifier = Classifier(indexer)
clusterer  = Clusterer(indexer)
evaluator  = Evaluator()
recommender = Recommender(indexer)
uploader   = Uploader(indexer)

print("=" * 55)
print("  All modules initialized successfully!")
print("  Open http://127.0.0.1:5000 in your browser")
print("=" * 55)

# ------------------------------------------------------------------ #
#  ROUTES                                                              #
# ------------------------------------------------------------------ #

@app.route('/')
def index():
    """Main search page."""
    stats = indexer.get_index_stats()
    return render_template('index.html', stats=stats)


@app.route('/search', methods=['POST'])
def search():
    """
    Main VSM / BM25 search endpoint.
    Accepts: { query, model, top_k }
    Returns: ranked list of matching notes with scores.
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    model = data.get('model', 'vsm')
    top_k = int(data.get('top_k', 10))

    if not query:
        return jsonify({'results': [], 'query_terms': [], 'error': 'Empty query'})

    # VSM or BM25 retrieval
    results = retriever.search(query, top_k=top_k, model=model)
    query_terms = retriever.get_query_terms(query)

    # Log interaction for recommender
    if results:
        recommender.log_interaction('default_user', results[0]['id'])

    # Get recommendations based on query
    recommendations = recommender.recommend_by_query(query, top_n=3)

    return jsonify({
        'results': results,
        'query_terms': [{'term': t, 'weight': round(w, 4)} for t, w in query_terms[:8]],
        'recommendations': recommendations,
        'total': len(results),
        'model_used': model.upper()
    })


@app.route('/boolean_search', methods=['POST'])
def boolean_search():
    """
    Boolean retrieval endpoint.
    Accepts: { query }
    Supports: AND, OR, NOT operators
    """
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'results': [], 'explanation': []})

    results = boolean.retrieve_with_docs(query)
    explanation = boolean.explain_query(query)

    return jsonify({
        'results': results,
        'explanation': explanation,
        'total': len(results)
    })


@app.route('/language_model_search', methods=['POST'])
def language_model_search():
    """
    Language Model retrieval endpoint.
    Accepts: { query, smoothing }
    Supports: jelinek_mercer, laplace, dirichlet smoothing
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    smoothing = data.get('smoothing', 'jelinek_mercer')

    if not query:
        return jsonify({'results': []})

    lang_model.smoothing = smoothing
    results = lang_model.retrieve_with_docs(query, top_k=8)

    return jsonify({
        'results': results,
        'smoothing': smoothing,
        'total': len(results)
    })


@app.route('/classify', methods=['POST'])
def classify():
    """
    Text classification endpoint.
    Accepts: { text, method }
    Returns: predicted unit label using KNN, Naive Bayes, SVM
    """
    data = request.get_json()
    text = data.get('text', '').strip()
    method = data.get('method', 'all')

    if not text:
        return jsonify({'error': 'No text provided'})

    result = classifier.classify(text, method=method)
    top_terms = classifier.get_top_terms_per_class(top_n=5)

    return jsonify({
        'classification': result,
        'top_terms_per_unit': top_terms
    })


@app.route('/cluster', methods=['POST'])
def cluster():
    """
    Clustering endpoint.
    Accepts: { method, k }
    Returns: cluster assignments with top terms
    """
    data = request.get_json()
    method = data.get('method', 'kmeans')
    k = int(data.get('k', 5))

    if method == 'hierarchical':
        result = clusterer.hierarchical_cluster(n_clusters=k)
    else:
        result = clusterer.kmeans_cluster(k=k)

    return jsonify(result)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluation endpoint.
    Accepts: { query, relevant_ids }
    Returns: Precision, Recall, F1, MAP, NDCG scores
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    relevant_ids = data.get('relevant_ids', [])

    if not query:
        return jsonify({'error': 'No query provided'})

    # Retrieve results using VSM
    vsm_results = retriever.search(query, top_k=10, model='vsm')
    retrieved_ids = [r['id'] for r in vsm_results]

    if not relevant_ids:
        # Auto-mark top-3 as relevant if none provided (for demo)
        relevant_ids = retrieved_ids[:3]

    # Full evaluation
    eval_report = evaluator.evaluate(retrieved_ids, relevant_ids, retrieved_ids)
    pr_curve = evaluator.precision_recall_curve(retrieved_ids, relevant_ids)
    interpolated = evaluator.interpolated_precision(pr_curve)

    return jsonify({
        'metrics': eval_report,
        'pr_curve': pr_curve,
        'interpolated_pr': interpolated,
        'retrieved_ids': retrieved_ids,
        'relevant_ids': relevant_ids
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommendation endpoint.
    Accepts: { doc_id, query, mode }
    mode: content | collaborative | hybrid | unit
    """
    data = request.get_json()
    doc_id = data.get('doc_id')
    query = data.get('query', '')
    mode = data.get('mode', 'content')
    unit = data.get('unit', '')

    if mode == 'collaborative':
        recs = recommender.collaborative_recommend('default_user', top_n=5)
    elif mode == 'hybrid':
        recs = recommender.hybrid_recommend(query, top_n=5)
    elif mode == 'unit' and unit:
        recs = recommender.get_unit_recommendations(unit, top_n=5)
    elif doc_id:
        recs = recommender.get_similar_docs(doc_id, top_n=5)
    else:
        recs = recommender.recommend_by_query(query, top_n=5)

    return jsonify({'recommendations': recs})


@app.route('/index_stats')
def index_stats():
    """Returns index statistics."""
    stats = indexer.get_index_stats()
    return jsonify(stats)


@app.route('/documents')
def all_documents():
    """Returns all documents in the collection."""
    docs = [
        {
            'id': doc['id'],
            'topic': doc['topic'],
            'unit': doc['unit'],
            'content_preview': doc['content'][:150] + '...'
        }
        for doc in indexer.documents
    ]
    return jsonify({'documents': docs, 'total': len(docs)})


@app.route('/document/<int:doc_id>')
def get_document(doc_id):
    """Returns full document by ID."""
    doc = indexer.get_doc_by_id(doc_id)
    if doc:
        similar = recommender.get_similar_docs(doc_id, top_n=4)
        return jsonify({'document': doc, 'similar': similar})
    return jsonify({'error': 'Document not found'}), 404


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    File upload endpoint.
    Accepts: multipart/form-data with 'file' field + optional 'subject' field.
    Extracts text from PDF/PPTX, preprocesses, and injects into live index.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    subject = request.form.get('subject', 'Uploaded Notes').strip() or 'Uploaded Notes'

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF and PPTX files are supported'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        created_docs = uploader.process_file(save_path, filename, subject)
        stats = indexer.get_index_stats()
        return jsonify({
            'success': True,
            'filename': filename,
            'sections_extracted': len(created_docs),
            'subject': subject,
            'docs': [
                {
                    'id': d['id'],
                    'topic': d['topic'],
                    'unit': d['unit'],
                    'content_preview': d['content'][:150] + '...' if len(d['content']) > 150 else d['content'],
                    'source': d.get('source', ''),
                    'page': d.get('page') or d.get('slide')
                }
                for d in created_docs
            ],
            'new_index_stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploaded_docs')
def uploaded_docs():
    """Returns all documents that came from uploaded files."""
    docs = uploader.get_uploaded_docs()
    return jsonify({
        'documents': [
            {
                'id': d['id'],
                'topic': d['topic'],
                'unit': d['unit'],
                'source': d.get('source', ''),
                'source_type': d.get('source_type', ''),
                'page': d.get('page') or d.get('slide'),
                'content_preview': d['content'][:150] + '...' if len(d['content']) > 150 else d['content']
            }
            for d in docs
        ],
        'upload_history': uploader.get_upload_history()
    })


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true', port=5000)