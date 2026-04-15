from flask import Flask, redirect, session, request, jsonify
from flask_cors import CORS
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from huggingface_hub import snapshot_download
from lime.lime_text import LimeTextExplainer
from pipeline import run_full_pipeline          # import pipeline
 
import os
import gc
import base64
import json
import secrets
import hashlib
import base64 as b64
import threading
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from bs4 import BeautifulSoup
 
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import login as hf_login
 
# Login to HF Hub if token available
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    hf_login(token=hf_token)
 
# ==========================================================
# CONFIG
# ==========================================================
 
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.readonly"
]
 
PROJECT_ID = "email-detection-42c8f"
PUBSUB_TOPIC = f"projects/{PROJECT_ID}/topics/gmail-topic"
 
HF_REPO = "littleprophisher/phishing-bert"
MODEL_PATH = "bert_model"
 
# Lock to prevent duplicate email processing
processing_lock = threading.Lock()
 
# Lock to prevent multiple retraining runs simultaneously
retraining_lock = threading.Lock()
 
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
 
CORS(app)
 
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
 
# ==========================================================
# FIREBASE INIT
# ==========================================================
 
try:
    firebase_key_json = os.environ.get("FIREBASE_KEY")
    if firebase_key_json:
        print("Using FIREBASE_KEY from environment")
        key_dict = json.loads(firebase_key_json)
        cred = credentials.Certificate(key_dict)
    else:
        print("Using firebase_key.json file")
        cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase init error: {e}")
    raise
 
# ==========================================================
# BACKGROUND BERT DOWNLOAD
# ==========================================================
 
model_ready = False
 
def background_download():
    global model_ready
    try:
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            print(f"Background: Downloading BERT from HF Hub: {HF_REPO}")
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print("Background: BERT downloaded successfully")
        else:
            print("Background: BERT already exists locally")
        model_ready = True
    except Exception as e:
        print(f"Background download error: {e}")
 
threading.Thread(target=background_download, daemon=True).start()
 
# ==========================================================
# LOAD MODEL
# Use lists so pipeline.py can update the reference
# ==========================================================
 
# Wrapped in list so pipeline can update the reference
_model = [None]
_tokenizer = [None]
 
def load_model():
    if _tokenizer[0] is None or _model[0] is None:
 
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            print("Waiting for BERT download...")
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False
            )
 
        print("Loading BERT model into memory...")
        _tokenizer[0] = BertTokenizer.from_pretrained(MODEL_PATH)
        _model[0] = BertForSequenceClassification.from_pretrained(
            MODEL_PATH
        )
        _model[0].eval()
        _model[0].to("cpu")
        print("Model loaded successfully")
 
# Convenience accessors
def get_model():
    return _model[0]
 
def get_tokenizer():
    return _tokenizer[0]
 
# ==========================================================
# BERT DETECTION
# ==========================================================
 
def detect_email(text):
    try:
        load_model()
        print("Running BERT inference...")
 
        inputs = get_tokenizer()(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        print("Tokenized successfully")
 
        with torch.no_grad():
            outputs = get_model()(**inputs)
        print("Inference done")
 
        probs = F.softmax(outputs.logits.float(), dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        label = "Phishing" if predicted.item() == 1 else "Legitimate"
 
        del inputs, outputs, probs
        gc.collect()
 
        return label, confidence.item()
 
    except Exception as e:
        print(f"detect_email error: {e}")
        import traceback
        traceback.print_exc()
        return "Legitimate", 0.5
 
# ==========================================================
# LIME EXPLANATION
# ==========================================================
 
def get_lime_explanation(text):
    try:
        load_model()
        print("LIME: starting explainer...")
 
        explainer = LimeTextExplainer(
            class_names=["Legitimate", "Phishing"]
        )
 
        def predict_proba(texts):
            all_probs = []
            for t in texts:
                inputs = get_tokenizer()(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )
                with torch.no_grad():
                    outputs = get_model()(**inputs)
                probs = F.softmax(outputs.logits.float(), dim=1)
                all_probs.append(
                    probs.squeeze().detach().float().numpy()
                )
                del inputs, outputs, probs
                gc.collect()
            return np.array(all_probs, dtype=np.float64)
 
        print("LIME: running explain_instance...")
        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=50
        )
 
        result = {
            word: round(float(weight), 4)
            for word, weight in explanation.as_list()
        }
 
        print(f"LIME done: {result}")
        return result
 
    except Exception as e:
        print(f"LIME error: {e}")
        import traceback
        traceback.print_exc()
        return None
 
# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
 
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)
 
 
def get_email_body(payload):
 
    if "parts" in payload:
        for part in payload["parts"]:
 
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8")
 
            if part["mimeType"] == "text/html":
                data = part["body"].get("data")
                if data:
                    html = base64.urlsafe_b64decode(data).decode("utf-8")
                    return clean_html(html)
 
    data = payload.get("body", {}).get("data")
    if data:
        return base64.urlsafe_b64decode(data).decode("utf-8")
 
    return ""
 
 
def get_gmail_service():
 
    doc = db.collection("gmail_tokens").document("user").get()
 
    if not doc.exists:
        return None
 
    creds_dict = doc.to_dict()
 
    creds = Credentials(
        token=creds_dict["token"],
        refresh_token=creds_dict["refresh_token"],
        token_uri=creds_dict["token_uri"],
        client_id=creds_dict["client_id"],
        client_secret=creds_dict["client_secret"],
        scopes=creds_dict["scopes"]
    )
 
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        db.collection("gmail_tokens").document("user").update({
            "token": creds.token
        })
 
    return build("gmail", "v1", credentials=creds)
 
# ============================
# EMAIL PROCESSING — real-time
# ============================
 
def process_latest_email():
 
    if not processing_lock.acquire(blocking=False):
        print("Already processing — skipping duplicate")
        return
 
    try:
        service = get_gmail_service()
 
        if not service:
            print("No Gmail token")
            return
 
        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=5
        ).execute()
 
        messages = results.get("messages", [])
 
        if not messages:
            print("No messages found")
            return
 
        for message in messages:
 
            msg_id = message["id"]
 
            if db.collection("emails").document(msg_id).get().exists:
                print(f"Already processed: {msg_id}")
                continue
 
            msg = service.users().messages().get(
                userId="me",
                id=msg_id,
                format="full"
            ).execute()
 
            payload = msg.get("payload", {})
            headers = payload.get("headers", [])
 
            subject = ""
            for header in headers:
                if header["name"] == "Subject":
                    subject = header["value"]
                    break
 
            body = get_email_body(payload)
            combined_text = (subject + " " + body)[:1000]
 
            # Step 1 — BERT
            print(f"Running BERT on: {subject}")
            label, confidence = detect_email(combined_text)
            print(f"BERT result: {label} ({confidence:.2f})")
 
            # Step 2 — Save immediately
            db.collection("emails").document(msg_id).set({
                "email_id": msg_id,
                "subject": subject,
                "content": body,
                "prediction": label,
                "confidence": confidence,
                "lime_explanation": None
            })
            print(f"Saved to Firebase: {subject}")
 
            # Step 3 — Low confidence queue
            if confidence < 0.95:
                db.collection("low_confidence").document(msg_id).set({
                    "email_id": msg_id,
                    "subject": subject,
                    "content": body,
                    "prediction": label,
                    "confidence": confidence,
                    "feedback_given": False
                })
                print(f"Added to low_confidence: {subject}")
 
            # Step 4 — LIME
            print(f"Running LIME on: {subject}")
            lime_explanation = get_lime_explanation(combined_text)
 
            if lime_explanation:
                db.collection("emails").document(msg_id).update({
                    "lime_explanation": lime_explanation
                })
                print(f"LIME saved: {subject}")
            else:
                print(f"LIME failed — email saved without explanation")
 
            print(f"Email fully processed: {subject}")
            break
 
    except Exception as e:
        print(f"Email processing error: {e}")
        import traceback
        traceback.print_exc()
 
    finally:
        processing_lock.release()
 
# ==========================================================
# ROUTES
# ==========================================================
 
@app.route("/")
def home():
    return "AI Phishing Email Detector Running"
 
 
@app.route("/health")
def health():
    return "OK"
 
 
@app.route("/status")
def status():
    return jsonify({
        "flask": "running",
        "bert_downloaded": os.path.exists(MODEL_PATH) and bool(os.listdir(MODEL_PATH)),
        "bert_loaded": _model[0] is not None,
        "retraining_running": retraining_lock.locked()
    })
 
 
@app.route("/login")
def login():
    try:
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = b64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
 
        flow = Flow.from_client_secrets_file(
            "credentials.json",
            scopes=SCOPES,
            redirect_uri=os.environ.get(
                "REDIRECT_URI",
                "https://littleprophisher-phishing-backend.hf.space/oauth2callback"
            )
        )
 
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )
 
        session["state"] = state
        session["code_verifier"] = code_verifier
        session.modified = True
        return redirect(authorization_url)
 
    except Exception as e:
        print(f"Login error: {e}")
        return f"Login error: {str(e)}", 500
 
 
@app.route("/oauth2callback")
def oauth2callback():
    try:
        state = session.get("state")
        code_verifier = session.get("code_verifier")
 
        if not state:
            return redirect("/login")
 
        from urllib.parse import urlparse, parse_qs
        import requests as req
 
        parsed = urlparse(request.url)
        code = parse_qs(parsed.query).get("code", [None])[0]
 
        if not code:
            return "No code in callback URL", 400
 
        with open("credentials.json") as f:
            creds_data = json.load(f)
 
        web = creds_data.get("web", creds_data.get("installed", {}))
 
        token_payload = {
            "code": code,
            "client_id": web["client_id"],
            "client_secret": web["client_secret"],
            "redirect_uri": os.environ.get(
                "REDIRECT_URI",
                "https://littleprophisher-phishing-backend.hf.space/oauth2callback"
            ),
            "grant_type": "authorization_code"
        }
 
        if code_verifier:
            token_payload["code_verifier"] = code_verifier
 
        token_response = req.post(
            "https://oauth2.googleapis.com/token",
            data=token_payload
        )
 
        token_data = token_response.json()
 
        if "error" in token_data:
            return f"Token error: {token_data}", 500
 
        db.collection("gmail_tokens").document("user").set({
            "token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token", ""),
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": web["client_id"],
            "client_secret": web["client_secret"],
            "scopes": SCOPES
        })
 
        return "Gmail Connected Successfully! You can close this tab."
 
    except Exception as e:
        print(f"OAuth error: {e}")
        return f"OAuth error: {str(e)} — <a href='/login'>Click here to try again</a>", 500
 
 
@app.route("/start-watch")
def start_watch():
    try:
        service = get_gmail_service()
 
        if not service:
            return jsonify({"error": "Gmail not connected"}), 400
 
        watch_request = {
            "labelIds": ["INBOX"],
            "topicName": PUBSUB_TOPIC
        }
 
        response = service.users().watch(
            userId="me",
            body=watch_request
        ).execute()
 
        print("Gmail watch started:", response)
 
        return jsonify({
            "message": "Gmail watch started",
            "response": response
        })
 
    except Exception as e:
        print("Watch start error:", e)
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/gmail/webhook", methods=["POST"])
def gmail_webhook():
 
    envelope = request.get_json()
 
    if not envelope or "message" not in envelope:
        return ("Bad Request", 400)
 
    print("Gmail push received")
 
    threading.Thread(
        target=process_latest_email,
        daemon=True
    ).start()
 
    return ("", 200)
 
 
# ==========================================================
# API ROUTES
# ==========================================================
 
@app.route("/api/emails")
def get_emails():
    docs = db.collection("emails").stream()
    emails = [doc.to_dict() for doc in docs]
    return jsonify(emails)
 
 
@app.route("/api/low-confidence")
def get_low_confidence():
    docs = db.collection("low_confidence").stream()
    emails = []
    for doc in docs:
        data = doc.to_dict()
        if data.get("feedback_given") == False:
            emails.append(data)
    return jsonify(emails)
 
 
@app.route("/api/submit-feedback", methods=["POST"])
def submit_feedback():
 
    data = request.json
    email_id = data["email_id"]
    user_label = data["user_label"]
 
    doc = db.collection("low_confidence").document(email_id).get()
 
    if not doc.exists:
        return jsonify({"error": "Email not found"}), 404
 
    email_data = doc.to_dict()
 
    db.collection("user_feedback").add({
        "email_id": email_id,
        "subject": email_data["subject"],
        "content": email_data["content"],
        "user_label": user_label,
        "model_prediction": email_data["prediction"],
        "confidence": email_data["confidence"],
        "generation_status": "pending"
    })
 
    db.collection("low_confidence").document(email_id).update({
        "feedback_given": True
    })
 
    return jsonify({"message": "Feedback stored"})
 
 
# ==========================================================
# TRIGGER RETRAINING — called by Firebase Cloud Function
# Completely separate from real-time email flow
# ==========================================================
 
@app.route("/trigger-retraining", methods=["POST"])
def trigger_retraining():
 
    # Don't run two retraining jobs at once
    if retraining_lock.locked():
        return jsonify({
            "message": "Retraining already in progress"
        }), 200
 
    def run_pipeline_with_lock():
        with retraining_lock:
            run_full_pipeline(
                db=db,
                bert_model_ref=_model,
                bert_tokenizer_ref=_tokenizer,
                load_model_fn=load_model,
                HF_REPO=HF_REPO
            )
 
    threading.Thread(
        target=run_pipeline_with_lock,
        daemon=True
    ).start()
 
    return jsonify({
        "message": "Pipeline started — will retrain only if threshold of 20 emails per class is met"
    })
 
@app.route("/retraining-status")
def retraining_status():
    return jsonify({
        "retraining_running": retraining_lock.locked()
    })
 
 
# ==========================================================
# RELOAD MODEL — called manually if needed
# ==========================================================
 
@app.route("/reload-model", methods=["POST"])
def reload_model():
    global _model, _tokenizer
 
    try:
        print("Reloading model from HuggingFace Hub...")
 
        _model[0] = None
        _tokenizer[0] = None
        gc.collect()
 
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False,
            force_download=True
        )
 
        load_model()
 
        print("Model reloaded successfully")
        return jsonify({"message": "Model reloaded successfully"})
 
    except Exception as e:
        print("Reload error:", e)
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/api/memory")
def memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return jsonify({
        "used_mb": round(mem, 2),
        "status": "ok" if mem < 14000 else "warning"
    })
 
 
@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": str(e)}), 500
 
 
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404
 
 
# ==========================================================
# RUN — port 7860 for HuggingFace Spaces
# ==========================================================
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))