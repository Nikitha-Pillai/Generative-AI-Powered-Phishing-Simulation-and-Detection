import os
import gc
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
# GENERATE ONE SIMILAR EMAIL
# Low temperature = stays close to original wording
# ==========================================================

def generate_one_similar_email(llm_model, llm_tokenizer, label, subject, content, attempt=1):
    """
    Generates one email very similar to original.
    Low temperature (0.3-0.55) keeps it close to original.
    Small increase per attempt creates slight variation
    to avoid duplicates while staying similar.
    """

    # LOW temperature = stays similar to original
    # Small increase per attempt = slight variation, avoids duplicates
    temperature = round(0.3 + (attempt * 0.05), 2)
    temperature = min(temperature, 0.55)  # never go above 0.55

    # Rewrite prompt — tells TinyLlama to rephrase not reinvent
    prompt = f"""Rewrite the following {label} email.
Keep the exact same topic, meaning and structure.
Only change the wording and phrasing of each sentence.
Keep Subject, greeting, body and closing signature.

Original email:
Subject: {subject}
{content[:250]}

Rewritten email with same meaning but different words:
Subject:"""

    try:
        inputs = llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=450
        )

        with torch.no_grad():
            output = llm_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=temperature,
                top_p=0.85,
                repetition_penalty=1.15
            )

        full_text = llm_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # Extract only generated part after prompt
        generated = full_text[len(prompt):].strip()

        # Fallback if extraction failed
        if len(generated) < 50:
            generated = full_text.replace(prompt, "").strip()

        return generated

    except Exception as e:
        print(f"PIPELINE: Generation error attempt {attempt}: {e}")
        return ""

# ==========================================================
# GENERATE 5 UNIQUE SIMILAR EMAILS
# ==========================================================

def generate_unique_similar_emails(llm_model, llm_tokenizer, label, subject, content, num=5):
    """
    Generates num emails similar to original.
    - Low temperature keeps them similar to original
    - Fingerprint check ensures no duplicates
    - Relaxed validation ensures more emails pass
    """

    results = []
    seen_fingerprints = set()
    max_attempts = num * 5  # try up to 25 times to get 5

    for attempt in range(1, max_attempts + 1):

        if len(results) >= num:
            break

        print(f"PIPELINE: Attempt {attempt} (have {len(results)}/{num})...")

        generated = generate_one_similar_email(
            llm_model, llm_tokenizer,
            label, subject, content,
            attempt=attempt
        )

        # Check length only — relaxed validation
        if not is_valid_email(generated):
            print(f"PIPELINE: Attempt {attempt} — too short, skipping")
            continue

        # Check duplicate using first 80 chars as fingerprint
        fingerprint = generated[:80].lower().strip()
        if fingerprint in seen_fingerprints:
            print(f"PIPELINE: Attempt {attempt} — duplicate, skipping")
            continue

        # Check not identical to original
        original_fp = content[:80].lower().strip()
        if fingerprint == original_fp:
            print(f"PIPELINE: Attempt {attempt} — too identical to original, skipping")
            continue

        # Valid unique similar email
        seen_fingerprints.add(fingerprint)
        results.append(generated)
        print(f"PIPELINE: Got similar email {len(results)}/{num} ✓")

    print(f"PIPELINE: Generated {len(results)} similar unique emails")
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

    print("=" * 50)
    print("PIPELINE: Starting full retraining pipeline")
    print("=" * 50)

    llm_model = None
    llm_tokenizer = None

    try:

        # --------------------------------------------------
        # STEP 1 — Check for pending feedback
        # --------------------------------------------------
        print("PIPELINE: Checking for pending feedback...")

        pending_docs = db.collection("user_feedback") \
                         .where("generation_status", "==", "pending") \
                         .stream()

        pending = list(pending_docs)
        print(f"PIPELINE: Found {len(pending)} pending feedback emails")

        if not pending:
            print("PIPELINE: Nothing to process — exiting")
            return

        # --------------------------------------------------
        # STEP 2 — Load TinyLlama
        # --------------------------------------------------
        print("PIPELINE: Loading TinyLlama...")

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
        print("PIPELINE: TinyLlama loaded")

        # --------------------------------------------------
        # STEP 3 — Generate 5 similar emails per feedback
        # --------------------------------------------------
        training_data = []

        for doc in pending:
            data = doc.to_dict()
            label = data.get("user_label", "Legitimate")
            subject = data.get("subject", "")
            content = data.get("content", "")[:300]

            print(f"PIPELINE: Generating 5 similar emails for — {subject[:50]}")

            unique_emails = generate_unique_similar_emails(
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                label=label,
                subject=subject,
                content=content,
                num=5
            )

            # Store each valid email in Firebase
            saved_count = 0
            for i, email_text in enumerate(unique_emails):

                # Final validation before saving
                if not is_valid_email(email_text):
                    print(f"PIPELINE: Email {i+1} failed final validation — skipping")
                    continue

                # Extract subject line if present
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

                training_data.append({
                    "text": email_subject + " " + email_body,
                    "label": 1 if label == "Phishing" else 0
                })

                saved_count += 1
                print(f"PIPELINE: Saved similar email {saved_count} to Firebase")

            # Mark this feedback as completed
            doc.reference.update({
                "generation_status": "completed"
            })
            print(f"PIPELINE: Marked feedback completed — {subject[:50]}")

        print(f"PIPELINE: Total training emails: {len(training_data)}")

        # --------------------------------------------------
        # Free TinyLlama before retraining
        # --------------------------------------------------
        del llm_model, llm_tokenizer
        llm_model = None
        llm_tokenizer = None
        gc.collect()
        print("PIPELINE: TinyLlama freed from memory")

        if not training_data:
            print("PIPELINE: No valid training data — skipping retraining")
            return

        # --------------------------------------------------
        # STEP 4 — Retrain BERT
        # --------------------------------------------------
        print("PIPELINE: Starting BERT retraining...")

        load_model_fn()

        bert_model = bert_model_ref[0]
        bert_tokenizer = bert_tokenizer_ref[0]

        dataset = EmailDataset(training_data, bert_tokenizer)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

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
            print(f"PIPELINE: Epoch {epoch+1}/3 — loss: {avg_loss:.4f}")

        bert_model.eval()
        print("PIPELINE: BERT retraining complete")

        # Free training RAM before push
        del optimizer, loader, dataset
        gc.collect()
        print("PIPELINE: Training RAM freed")

        # --------------------------------------------------
        # STEP 5 — Push retrained BERT to HF Hub
        # --------------------------------------------------
        try:
            print(f"PIPELINE: Pushing new BERT to HF Hub: {HF_REPO}")
            bert_model.push_to_hub(HF_REPO)
            bert_tokenizer.push_to_hub(HF_REPO)
            print("PIPELINE: BERT pushed to HF Hub successfully")
        except Exception as push_error:
            print(f"PIPELINE: Push error: {push_error}")
            traceback.print_exc()
            return

        # --------------------------------------------------
        # STEP 6 — Mark training emails as used
        # --------------------------------------------------
        try:
            unused_docs = db.collection("training_emails") \
                            .where("training_status", "==", "unused") \
                            .stream()
            for doc in unused_docs:
                doc.reference.update({"training_status": "used"})
            print("PIPELINE: Training emails marked as used")
        except Exception as e:
            print(f"PIPELINE: Mark used error: {e}")

        # --------------------------------------------------
        # STEP 7 — Update in-memory model references
        # --------------------------------------------------
        bert_model_ref[0] = bert_model
        bert_tokenizer_ref[0] = bert_tokenizer
        print("PIPELINE: In-memory BERT updated")

        # --------------------------------------------------
        # STEP 8 — Auto reload Flask model
        # No manual step needed ever again
        # --------------------------------------------------
        try:
            print("PIPELINE: Auto reloading Flask model...")
            response = requests.post(
                "https://littleprophisher-phishing-backend.hf.space/reload-model",
                timeout=30
            )
            print(f"PIPELINE: Reload response: {response.json()}")
        except Exception as e:
            print(f"PIPELINE: Auto reload error: {e}")

        print("=" * 50)
        print("PIPELINE: Complete! New BERT is live for real-time detection")
        print("=" * 50)

    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        traceback.print_exc()

    finally:
        if llm_model is not None:
            del llm_model
        if llm_tokenizer is not None:
            del llm_tokenizer
        gc.collect()
        print("PIPELINE: Cleanup done")