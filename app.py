from flask import Flask, render_template, request, jsonify
import json
import random
import re
import os
from difflib import SequenceMatcher

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ---------- Load data ----------
with open("data/faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

intents = faq_data.get("intents", [])
fallbacks = faq_data.get("fallbacks", [])
synonyms = faq_data.get("synonyms", {})

# Build training data from intents: use natural examples as training utterances
train_texts = []
train_labels = []
intent_map = {}  # id -> intent object for quick lookup
for intent in intents:
    iid = intent["id"]
    intent_map[iid] = intent
    examples = intent.get("examples", []) or intent.get("keywords", [])
    for ex in examples:
        train_texts.append(ex)
        train_labels.append(iid)
    # small augmentation: add variants (if relevant)
    for ex in examples:
        if "order" in ex and "my" not in ex:
            train_texts.append(ex.replace("order", "my order"))
            train_labels.append(iid)

if not train_texts:
    print("Warning: no training examples found in data/faqs.json")

# ---------- Helpers ----------
def clean(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def fuzzy_intent_within(user_q, allowed_intents):
    best_score = 0.0
    best_intent = None
    uq = clean(user_q)
    for intent in allowed_intents:
        for kw in (intent.get("examples", []) + intent.get("priority_keywords", []) + intent.get("keywords", [])):
            if not kw:
                continue
            score = SequenceMatcher(None, uq, clean(kw)).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent["id"]
    return best_intent, best_score

def synonyms_intent_within(user_q, allowed_intents):
    uq = user_q.lower()
    for term, intent_id in synonyms.items():
        if term in uq or intent_id in uq:
            # only return if this intent is allowed for the store
            for it in allowed_intents:
                if it["id"] == intent_id:
                    return intent_id
    return None

def get_allowed_intents_for_store(store):
    # store is a string like 'amazon' or 'global'
    allowed = []
    store = (store or "global").lower()
    for intent in intents:
        stores = [s.lower() for s in intent.get("stores", ["global"])]
        if store == "global" and ("global" in stores or True):
            # global selection should include everything
            allowed.append(intent)
        else:
            if "global" in stores or store in stores:
                allowed.append(intent)
    return allowed

def generate_openai_reply(user_q):
    key = os.getenv("OPENAI_API_KEY")
    if not key or not user_q or not user_q.strip():
        return None
    system_prompt = "You are a helpful support assistant for a college's customer portal. Answer briefly and accurately."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_q},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        try:
            import openai
            openai.api_key = key
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_q},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            content = resp["choices"][0]["message"]["content"]
            return content.strip() if content else None
        except Exception:
            return None

# ---------- Train a tiny classifier (on cleaned examples) ----------
clean_train_texts = [clean(t) for t in train_texts]
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(clean_train_texts)
clf = LogisticRegression(max_iter=400)
clf.fit(X_train, train_labels)
print("Intent classifier trained on", len(train_texts), "examples.")

# ---------- Main response function ----------
def get_response(user_question, store="global"):
    user_q = user_question or ""
    uq_clean = clean(user_q)
    store = (store or "global").lower()

    # get intents allowed for this store
    allowed_intents = get_allowed_intents_for_store(store)

    # 0) Priority keywords check within allowed intents first (exact phrase presence)
    for intent in allowed_intents:
        for pk in intent.get("priority_keywords", []):
            if not pk:
                continue
            if clean(pk) in uq_clean:
                return random.choice(intent.get("responses", []))

    # 1) Model prediction (on cleaned text) - if it predicts an intent outside allowed, we ignore it
    x = vectorizer.transform([uq_clean])
    probs = clf.predict_proba(x)[0]
    sorted_idx = probs.argsort()[::-1]
    top_idx = sorted_idx[0]
    top_label = clf.classes_[top_idx]
    top_prob = probs[top_idx]
    second_prob = probs[sorted_idx[1]] if len(probs) > 1 else 0.0

    # if predicted is allowed and confident, use it
    if top_label in intent_map and any(it["id"] == top_label for it in allowed_intents):
        if top_prob >= 0.65 and (top_prob - second_prob) >= 0.12:
            intent = intent_map.get(top_label)
            if intent:
                return random.choice(intent.get("responses", []))

    # 2) synonyms check within allowed intents
    syn_intent = synonyms_intent_within(user_q, allowed_intents)
    if syn_intent and syn_intent in intent_map:
        return random.choice(intent_map[syn_intent].get("responses", []))

    # 3) fuzzy matching on examples/keywords but limited to allowed intents
    best_intent, score = fuzzy_intent_within(user_q, allowed_intents)
    if best_intent and score >= 0.60:
        return random.choice(intent_map[best_intent].get("responses", []))

    # 4) simple substring search within allowed intents
    for intent in allowed_intents:
        for kw in (intent.get("examples", []) + intent.get("priority_keywords", []) + intent.get("keywords", [])):
            if kw and clean(kw) in uq_clean:
                return random.choice(intent.get("responses", []))

    # 5) If store-limited search fails and store != global, try global intents as fallback
    if store != "global":
        global_allowed = get_allowed_intents_for_store("global")
        # repeat fuzzy + substring on global set
        best_intent, score = fuzzy_intent_within(user_q, global_allowed)
        if best_intent and score >= 0.60:
            return random.choice(intent_map[best_intent].get("responses", []))
        for intent in global_allowed:
            for kw in (intent.get("examples", []) + intent.get("priority_keywords", []) + intent.get("keywords", [])):
                if kw and clean(kw) in uq_clean:
                    return random.choice(intent.get("responses", []))

    # 6) OpenAI fallback if configured
    ai_reply = generate_openai_reply(user_q)
    if ai_reply:
        return ai_reply

    # 7) default fallback responses
    return random.choice(fallbacks or ["Sorry, I don't know that."])

# ---------- Flask endpoints ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    user_message = data.get("message", "")
    store = (data.get("store") or "global").lower()
    reply = get_response(user_message, store=store)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
