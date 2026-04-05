import os
import gc
import sys
import torch
import requests
import traceback
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer,
    BertForSequenceClassification
)
from huggingface_hub import login as hf_login

# Force unbuffered output — logs appear immediately in HuggingFace
sys.stdout.reconfigure(line_buffering=True)

# ==========================================================
# EMAIL DATASET FOR BERT RETRAINING
# ==========================================================

class EmailDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

# ==========================================================
# VALIDATE EMAIL — relaxed, just check length
# ==========================================================

def is_valid_email(text):
    """Check email is long enough to be useful for training"""
    if not text or len(text.strip()) < 80:
        return False
    if len(text.split()) < 15:
        return False
    return True

# ==========================================================
# GENERATE BOTH CLASSES — 5 Phishing + 5 Legitimate
# per feedback email
# ==========================================================

def generate_both_classes(llm_model, llm_tokenizer, subject, content, num=5):
    """
    For every feedback email, generates:
    - 5 Phishing versions
    - 5 Legitimate versions
    This ensures BERT always trains on balanced data.
    Low temperature keeps emails realistic and grounded.
    """

    results = {"Phishing": [], "Legitimate": []}

    for target_label in ["Phishing", "Legitimate"]:

        seen_fingerprints = set()
        max_attempts = num * 5  # try up to 25 times to get 5

        for attempt in range(1, max_attempts + 1):

            if len(results[target_label]) >= num:
                break

            # LOW temperature = realistic, grounded output
            # Small increase per attempt = slight variation, avoids duplicates
            temperature = round(0.3 + (attempt * 0.05), 2)
            temperature = min(temperature, 0.55)

            if target_label == "Phishing":
                prompt = f"""Rewrite the following email as a convincing phishing email.
Keep the same topic and structure but make it sound suspicious.
Add urgency, threats, or fake links typical of phishing.
Keep Subject, greeting, body and closing signature.

Original email:
Subject: {subject}
{content[:250]}

Rewritten as phishing email:
Subject:"""

            else:
                prompt = f"""Rewrite the following email as a completely safe legitimate email.
Keep the same topic and structure but make it sound professional and trustworthy.
Remove any suspicious elements, urgency or threats.
Keep Subject, greeting, body and closing signature.

Original email:
Subject: {subject}
{content[:250]}

Rewritten as legitimate email:
Subject:"""

            try:
                print(f"PIPELINE: Starting generation — {target_label} attempt {attempt}, temp={temperature}", flush=True)

                inputs = llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=450
                )

                print(f"PIPELINE: Tokenized — {inputs['input_ids'].shape[1]} input tokens", flush=True)

                with torch.no_grad():
                    output = llm_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.85,
                        repetition_penalty=1.15
                    )

                print(f"PIPELINE: Generation done — {output.shape[1]} total tokens", flush=True)

                full_text = llm_tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )

                # Extract only the generated part after the prompt
                generated = full_text[len(prompt):].strip()

                # Fallback if extraction failed
                if len(generated) < 50:
                    generated = full_text.replace(prompt, "").strip()

                print(f"PIPELINE: Generated text preview: '{generated[:100]}'", flush=True)
                print(f"PIPELINE: Length: {len(generated.strip())} chars, Words: {len(generated.split())}", flush=True)

                # Relaxed length check
                if not is_valid_email(generated):
                    print(f"PIPELINE: FAILED validation — chars:{len(generated.strip())} words:{len(generated.split())}", flush=True)
                    continue

                # Duplicate check using first 80 chars as fingerprint
                fingerprint = generated[:80].lower().strip()
                if fingerprint in seen_fingerprints:
                    print(f"PIPELINE: FAILED duplicate check", flush=True)
                    continue

                seen_fingerprints.add(fingerprint)
                results[target_label].append(generated)
                print(f"PIPELINE: Got {target_label} email {len(results[target_label])}/{num} ✓", flush=True)

            except Exception as e:
                print(f"PIPELINE: Generation error {target_label} attempt {attempt}: {e}", flush=True)
                traceback.print_exc()
                continue

        print(f"PIPELINE: Generated {len(results[target_label])} {target_label} emails", flush=True)

    return results

# ==========================================================
# MAIN PIPELINE FUNCTION
# Called from app.py in background thread
# Does NOT affect real-time email flow
# ==========================================================

def run_full_pipeline(db, bert_model_ref, bert_tokenizer_ref, load_model_fn, HF_REPO):
    """
    db                 — Firestore client from app.py
    bert_model_ref     — list holding [model] so we can update it
    bert_tokenizer_ref — list holding [tokenizer]
    load_model_fn      — load_model() function from app.py
    HF_REPO            — HuggingFace repo string
    """

    # Minimum emails per class required before retraining BERT
    # 2 feedback emails x 5 generated each = 10 per class minimum
    # We use 20 per class = needs 4 feedback emails minimum
    MIN_EMAILS_PER_CLASS = 20

    print("=" * 50, flush=True)
    print("PIPELINE: Starting full retraining pipeline", flush=True)
    print("=" * 50, flush=True)

    llm_model = None
    llm_tokenizer = None

    try:

        # --------------------------------------------------
        # STEP 1 — Check for pending feedback
        # --------------------------------------------------
        print("PIPELINE: Checking for pending feedback...", flush=True)

        pending_docs = db.collection("user_feedback") \
                         .where("generation_status", "==", "pending") \
                         .stream()

        pending = list(pending_docs)
        print(f"PIPELINE: Found {len(pending)} pending feedback emails", flush=True)

        if not pending:
            print("PIPELINE: Nothing to process — exiting", flush=True)
            return

        # --------------------------------------------------
        # STEP 2 — Load TinyLlama
        # --------------------------------------------------
        print("PIPELINE: Loading TinyLlama...", flush=True)

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_login(token=hf_token)

        llm_tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype=torch.float16,
            trust_remote_code=True
        )
        llm_model.to("cpu")
        llm_model.eval()
        print("PIPELINE: TinyLlama loaded", flush=True)

        # --------------------------------------------------
        # STEP 3 — Generate 5 Phishing + 5 Legitimate per feedback
        # --------------------------------------------------
        new_training_data = []

        for doc in pending:
            data = doc.to_dict()
            subject = data.get("subject", "")
            content = data.get("content", "")[:300]

            print(f"PIPELINE: Generating 10 emails (5 Phishing + 5 Legitimate) for — {subject[:50]}", flush=True)

            both_classes = generate_both_classes(
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                subject=subject,
                content=content,
                num=5
            )

            # Save each generated email to Firebase training_emails collection
            for label, emails in both_classes.items():
                saved_count = 0

                for i, email_text in enumerate(emails):

                    # Final validation before saving
                    if not is_valid_email(email_text):
                        print(f"PIPELINE: {label} email {i+1} failed final validation — skipping", flush=True)
                        continue

                    # Extract subject line if present in generated text
                    lines = email_text.split("\n")
                    email_subject = subject
                    email_body = email_text

                    for line in lines[:3]:
                        if line.lower().startswith("subject:"):
                            email_subject = line.replace("Subject:", "").replace("subject:", "").strip()
                            email_body = "\n".join(lines[1:]).strip()
                            break

                    db.collection("training_emails").add({
                        "source_email_id": data.get("email_id", ""),
                        "subject": email_subject,
                        "content": email_body,
                        "label": label,
                        "training_status": "unused"
                    })

                    new_training_data.append({
                        "text": email_subject + " " + email_body,
                        "label": 1 if label == "Phishing" else 0
                    })

                    saved_count += 1
                    print(f"PIPELINE: Saved {label} email {saved_count} to Firebase", flush=True)

            # Mark this feedback doc as generation completed
            doc.reference.update({"generation_status": "completed"})
            print(f"PIPELINE: Marked feedback completed — {subject[:50]}", flush=True)

        print(f"PIPELINE: Newly generated emails this run: {len(new_training_data)}", flush=True)

        # --------------------------------------------------
        # Free TinyLlama before retraining
        # --------------------------------------------------
        del llm_model, llm_tokenizer
        llm_model = None
        llm_tokenizer = None
        gc.collect()
        print("PIPELINE: TinyLlama freed from memory", flush=True)

        # --------------------------------------------------
        # STEP 3.5 — Check minimum threshold before retraining
        # Count ALL unused emails in Firebase (across all runs)
        # --------------------------------------------------
        print("PIPELINE: Checking training threshold...", flush=True)

        unused_docs = list(
            db.collection("training_emails")
              .where("training_status", "==", "unused")
              .stream()
        )

        total_unused_phishing = sum(
            1 for d in unused_docs
            if d.to_dict().get("label") == "Phishing"
        )
        total_unused_legit = sum(
            1 for d in unused_docs
            if d.to_dict().get("label") == "Legitimate"
        )

        print(f"PIPELINE: Unused pool — Phishing: {total_unused_phishing}, Legit: {total_unused_legit}", flush=True)
        print(f"PIPELINE: Threshold required — {MIN_EMAILS_PER_CLASS} per class", flush=True)

        if total_unused_phishing < MIN_EMAILS_PER_CLASS or total_unused_legit < MIN_EMAILS_PER_CLASS:
            print(
                f"PIPELINE: Below threshold — skipping retrain. "
                f"Need {max(0, MIN_EMAILS_PER_CLASS - total_unused_phishing)} more Phishing, "
                f"{max(0, MIN_EMAILS_PER_CLASS - total_unused_legit)} more Legit emails.",
                flush=True
            )
            return

        # Build training data from ALL unused docs in Firebase
        training_data = []
        for d in unused_docs:
            data = d.to_dict()
            training_data.append({
                "text": data.get("subject", "") + " " + data.get("content", ""),
                "label": 1 if data.get("label") == "Phishing" else 0
            })

        print(f"PIPELINE: Training data from unused pool: {len(training_data)}", flush=True)

        # --------------------------------------------------
        #  Mix in anchor examples to prevent forgetting
        # Pull up to 20 per class from previously used training emails
        # --------------------------------------------------
        print("PIPELINE: Fetching anchor examples to prevent forgetting...", flush=True)

        anchor_data = []
        used_docs = db.collection("training_emails") \
                      .where("training_status", "==", "used") \
                      .stream()

        phishing_anchors = 0
        legit_anchors = 0
        MAX_ANCHORS_PER_CLASS = 20

        for d in used_docs:
            data = d.to_dict()
            label = data.get("label", "Legitimate")

            if label == "Phishing" and phishing_anchors < MAX_ANCHORS_PER_CLASS:
                anchor_data.append({
                    "text": data.get("subject", "") + " " + data.get("content", ""),
                    "label": 1
                })
                phishing_anchors += 1

            elif label == "Legitimate" and legit_anchors < MAX_ANCHORS_PER_CLASS:
                anchor_data.append({
                    "text": data.get("subject", "") + " " + data.get("content", ""),
                    "label": 0
                })
                legit_anchors += 1

        print(f"PIPELINE: Anchor examples — Phishing: {phishing_anchors}, Legit: {legit_anchors}", flush=True)

        # Combine new unused data + anchor examples
        final_training_data = training_data + anchor_data
        print(f"PIPELINE: Final training set — {len(final_training_data)} total emails", flush=True)

        # --------------------------------------------------
        # STEP 4 — Retrain BERT
        # --------------------------------------------------
        print("PIPELINE: Starting BERT retraining...", flush=True)

        load_model_fn()

        bert_model = bert_model_ref[0]
        bert_tokenizer = bert_tokenizer_ref[0]

        dataset = EmailDataset(final_training_data, bert_tokenizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(bert_model.parameters(), lr=1e-5)

        bert_model.train()

        for epoch in range(3):
            total_loss = 0
            for batch in loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["label"]

                optimizer.zero_grad()
                outputs = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()

            avg_loss = total_loss / len(loader)
            print(f"PIPELINE: Epoch {epoch+1}/3 — loss: {avg_loss:.4f}", flush=True)

        bert_model.eval()
        print("PIPELINE: BERT retraining complete", flush=True)

        # Free training RAM before push
        del optimizer, loader, dataset
        gc.collect()
        print("PIPELINE: Training RAM freed", flush=True)

        # --------------------------------------------------
        # STEP 5 — Push retrained BERT to HF Hub
        # --------------------------------------------------
        try:
            print(f"PIPELINE: Pushing new BERT to HF Hub: {HF_REPO}", flush=True)
            bert_model.push_to_hub(HF_REPO)
            bert_tokenizer.push_to_hub(HF_REPO)
            print("PIPELINE: BERT pushed to HF Hub successfully", flush=True)
        except Exception as push_error:
            print(f"PIPELINE: Push error: {push_error}", flush=True)
            traceback.print_exc()
            return

        # --------------------------------------------------
        # STEP 6 — Mark ALL unused training emails as used
        # --------------------------------------------------
        try:
            for doc in unused_docs:
                doc.reference.update({"training_status": "used"})
            print("PIPELINE: Training emails marked as used", flush=True)
        except Exception as e:
            print(f"PIPELINE: Mark used error: {e}", flush=True)

        # --------------------------------------------------
        # STEP 7 — Update in-memory model references
        # --------------------------------------------------
        bert_model_ref[0] = bert_model
        bert_tokenizer_ref[0] = bert_tokenizer
        print("PIPELINE: In-memory BERT updated", flush=True)

        # --------------------------------------------------
        # STEP 8 — Auto reload Flask model
        # --------------------------------------------------
        try:
            print("PIPELINE: Auto reloading Flask model...", flush=True)
            response = requests.post(
                "https://littleprophisher-phishing-backend.hf.space/reload-model",
                timeout=30
            )
            print(f"PIPELINE: Reload response: {response.json()}", flush=True)
        except Exception as e:
            print(f"PIPELINE: Auto reload error: {e}", flush=True)

        print("=" * 50, flush=True)
        print("PIPELINE: Complete! New BERT is live for real-time detection", flush=True)
        print("=" * 50, flush=True)

    except Exception as e:
        print(f"PIPELINE ERROR: {e}", flush=True)
        traceback.print_exc()

    finally:
        if llm_model is not None:
            del llm_model
        if llm_tokenizer is not None:
            del llm_tokenizer
        gc.collect()
        print("PIPELINE: Cleanup done", flush=True)