from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from newspaper import Article
from GoogleNews import GoogleNews
import torch.nn.functional as F
import os

# -------------------------
# App & Database Config
# -------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'veritas-secret-key'  # Security Key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -------------------------
# Database Model
# -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------
# AI Model Loading
# -------------------------
MODEL_PATH = "./model_output"
print(f"Loading AI model from {MODEL_PATH}...")

model = None
tokenizer = None
device = None
explainer = None

try:
    if not os.path.exists(MODEL_PATH):
        print("WARNING: Model folder not found. AI features will be disabled.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        explainer = SequenceClassificationExplainer(model, tokenizer)
        print(f"Veritas AI loaded on {device}!")
except Exception as e:
    print(f"Error loading model: {e}")

# -------------------------
# Helper Functions
# -------------------------
def fact_check_google(query):
    try:
        googlenews = GoogleNews()
        googlenews.set_lang('en')
        search_query = " ".join(query.split()[:12])
        googlenews.search(search_query)
        results = googlenews.result()
        
        if not results:
            return {"status": "No Results", "sources": []}
            
        reputable_domains = ['bbc', 'reuters', 'indiatoday', 'ndtv', 'timesofindia', 'cnn', 'nytimes', 'thehindu']
        found_sources = []
        
        for res in results[:5]:
            source = res.get('media', '').lower() or res.get('desc', '').lower()
            title = res.get('title', '')
            link = res.get('link', '#')
            is_reputable = any(dom in source or dom in link for dom in reputable_domains)
            found_sources.append({
                "title": title, "source": source if source else "Unknown",
                "link": link, "reputable": is_reputable
            })
            
        reputable_count = sum(1 for s in found_sources if s['reputable'])
        status = "Verified by Major Sources" if reputable_count >= 1 else ("Sources Found" if found_sources else "Unverified")
        return {"status": status, "sources": found_sources}
    except:
        return {"status": "Error Checking", "sources": []}

def predict_news(text):
    if not model: return "ERROR", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()
    return ("FAKE", fake_prob) if fake_prob > real_prob else ("REAL", real_prob)

# -------------------------
# Routes
# -------------------------
@app.route('/')
@login_required
def home():
    return render_template('index.html', name=current_user.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model: return jsonify({'error': 'Model not loaded.'})
    data = request.json
    text = data.get('text', '')
    url = data.get('url', '')
    final_text = text
    
    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            final_text = article.text
            if len(final_text) < 50: return jsonify({'error': 'URL content too short.'})
        except Exception as e:
            return jsonify({'error': f'Scraping error: {str(e)}'})

    if not final_text: return jsonify({'error': 'No text provided.'})

    label, confidence = predict_news(final_text)
    word_attributions = explainer(final_text)
    top_words = sorted(word_attributions, key=lambda x: abs(x[1]), reverse=True)[:8]
    fact_check = fact_check_google(final_text)
    
    verification_note = ""
    if label == "FAKE" and fact_check['status'] != "Verified by Major Sources":
        verification_note = "High Risk: AI detects fake patterns AND no major sources confirm this story."
    elif label == "FAKE" and fact_check['status'] == "Verified by Major Sources":
        verification_note = "Caution: AI flagged text style, but reputable sources are reporting this. Check context."
    elif label == "REAL" and fact_check['status'] == "Verified by Major Sources":
        verification_note = "Verified: Content seems authentic and is corroborated by major outlets."

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2),
        'preview': final_text[:200] + "...",
        'explanation': top_words,
        'fact_check': fact_check,
        'verification_note': verification_note
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True, port=5000)