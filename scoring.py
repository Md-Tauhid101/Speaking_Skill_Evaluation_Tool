# scoring.py
# Full scoring backend following rubric exactly (Option A)
# Requires: sentence-transformers, transformers, language-tool-python, nltk (for vader), pandas (optional)
# Run: pip install sentence-transformers transformers language-tool-python nltk

import string
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import language_tool_python
import math

# --- Config (use your existing dictionaries) ---
KEYWORDS = {
    # MUST HAVE
    "name": ["my name", "i am", "myself"],
    "age": ["years old"],
    "school_class": ["school", "class", "grade"],
    "family": ["family", "mother", "father", "parents"],
    "hobby_interest": ["hobby", "enjoy", "like to do", "play", "playing", "interest"],

    # GOOD TO HAVE
    "about_family_extra": ["special thing about my family", "my family is", "they are", "kind hearted", "soft spoken"],
    "origin_location": ["i am from", "my parents are from", "we are from"],
    "ambition_goal_dream": ["my goal", "my dream", "i want to", "my ambition"],
    "unique_fact": ["fun fact", "interesting thing", "something unique", "people don't know"],
    "strengths_achievements": ["strength", "achievement", "i am good at", "i have achieved"]
}

DESCRIPTIONS = {
    "content_structure":
        "The student should introduce themselves with name, age, class or school, family details, hobbies or interests, and a unique fact in a clear structure.",
    "salutation":
        "The introduction should begin with a polite salutation such as hello, good morning, or greetings.",
    "hobby":
        "The student should describe what they enjoy doing, their hobbies, interests or what they do in their free time.",
    "unique_fact":
        "The student should mention a unique, special, or interesting fact about themselves.",
    "family":
        "The student should briefly describe their family or household members.",
    "school":
        "The student should mention their class or school details as part of the self introduction.",
    "closing":
        "The introduction should end with a polite closing remark such as thank you for listening.",
    "engagement":
        "The overall tone of the introduction should feel positive, engaging, and enthusiastic."
}

WEIGHTS = {
    "content_structure": 40,
    "speech_rate": 10,
    "language_grammar": 20,
    "clarity": 15,
    "engagement": 15
}

# --- Utilities ---
def clean_text(text: str) -> str:
    t = text.lower().replace("\n", " ").strip()
    return " ".join(t.split())

# def remove_punctuation(text: str) -> str:
#     return text.translate(str.maketrans("", "", string.punctuation))

def word_count(text: str) -> int:
    return len(clean_text(text).split())

# --- Keyword / Rule functions (Steps 5.2 & 5.3) ---
def keyword_score_single_category(text: str, keywords: list) -> float:
    text = clean_text(text)
    found = 0
    for kw in keywords:
        if kw in text:
            found += 1
    return found / len(keywords) if len(keywords) > 0 else 0.0

def rule_category_presence(text: str, keyword_dict: dict) -> float:
    """Fraction of categories present at least once."""
    text = clean_text(text)
    total = len(keyword_dict)
    matched = 0
    for cat, kws in keyword_dict.items():
        if any(kw in text for kw in kws):
            matched += 1
    return matched / total if total > 0 else 0.0

def flow_score(text: str) -> float:
    # simplified order check (salutation -> name -> age -> closing)
    text = clean_text(text)
    order = ["hello", "my name", "years old", "thank you"]
    positions = [text.find(o) for o in order]
    observed = [p for p in positions if p != -1]
    # if found in the same relative order
    if observed == sorted(observed) and len(observed) >= 2:
        return 1.0
    return 0.0

# --- Semantic model (Step 6) ---
_model = None
def load_semantic_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def semantic_similarity(text: str, description: str) -> float:
    model = load_semantic_model()
    emb1 = model.encode(text, convert_to_tensor=True)
    emb2 = model.encode(description, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    # cosine similarity from sentence-transformers is usually >0 for related text; clamp 0-1
    return max(0.0, min(score, 1.0))

# --- Speech Rate mapping (use duration_sec, if unknown you must provide duration) ---
def compute_wpm(text: str, duration_sec: float) -> float:
    wc = word_count(text)
    return (wc / duration_sec) * 60.0

def speech_rate_score_from_wpm(wpm: float) -> float:
    # mapping to rubric points (out of 10)
    if wpm > 161:
        return 2.0
    elif 141 <= wpm <= 160:
        return 6.0
    elif 111 <= wpm <= 140:
        return 10.0
    elif 81 <= wpm <= 110:
        return 8.0
    else:
        return 2.0

# --- Grammar & TTR (Language & Grammar) ---
tool = language_tool_python.LanguageTool('en-US')

def grammar_score_points(text: str) -> float:
    words = max(1, word_count(text))
    errors = tool.check(text)
    error_count = len(errors)
    errors_per_100 = (error_count / words) * 100.0
    grammar_score = 1.0 - min(errors_per_100 / 10.0, 1.0)
    # map to rubric points out of 10
    if grammar_score > 0.9:
        return 10.0
    elif grammar_score >= 0.7:
        return 8.0
    elif grammar_score >= 0.5:
        return 6.0
    elif grammar_score >= 0.3:
        return 4.0
    else:
        return 2.0

def ttr_points(text: str) -> float:
    words = clean_text(text).split()
    if len(words) == 0:
        return 2.0
    ttr = len(set(words)) / len(words)
    # map to rubric out of 10
    if ttr >= 0.7:
        return 10.0
    elif ttr >= 0.5:
        return 8.0
    elif ttr >= 0.3:
        return 4.0
    else:
        return 2.0

# --- Clarity (filler rate) ---
FILLERS = ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"]

def filler_rate(text: str) -> float:
    text = clean_text(text)
    total = max(1, len(text.split()))
    filler_count = sum(text.count(f) for f in FILLERS)
    return (filler_count / total) * 100.0

def clarity_points(text: str) -> float:
    # rubric: lower filler rate -> higher points, we'll convert filler % to 0-1 then *15
    rate = filler_rate(text)
    # convert to score 0-1: 0% -> 1.0 ; >15% -> 0
    score = max(0.0, 1.0 - (rate / 15.0))
    return score * WEIGHTS["clarity"]

# --- Engagement (sentiment) ---
_sent_model = None
def load_sentiment_model():
    global _sent_model
    if _sent_model is None:
        _sent_model = pipeline("sentiment-analysis")
    return _sent_model

def engagement_points(text: str) -> float:
    model = load_sentiment_model()
    res = model(text[:512])[0]  # limit to first 512 chars to be safe
    label = res['label']
    score = res['score']
    # Convert label/score to 0-1 (positive higher)
    if label.upper().startswith("POS"):
        val = score
    else:
        val = 1.0 - score
    return val * WEIGHTS["engagement"]

# --- Combine signals for Content & Structure categories ---
def combined_signal(rule_score: float, semantic_score: float, keyword_score: float) -> float:
    return 0.4 * rule_score + 0.4 * semantic_score + 0.2 * keyword_score

def compute_content_structure_scores(transcript: str):
    t = clean_text(transcript)
    must_have = ["name", "age", "school_class", "family", "hobby_interest"]
    good_have = ["about_family_extra", "origin_location", "ambition_goal_dream", "unique_fact", "strengths_achievements"]

    per_category = {}
    raw_total = 0.0  # raw points out of 30 (5*4 + 5*2)
    # score must-have categories (each up to 4 points)
    for cat in must_have:
        kws = KEYWORDS.get(cat, [])
        rule = 1.0 if any(kw in t for kw in kws) else 0.0
        key = keyword_score_single_category(t, kws)
        # description for semantic: use specific if exists, else content_structure anchor
        desc = DESCRIPTIONS.get(cat, DESCRIPTIONS["content_structure"])
        sem = semantic_similarity(transcript, desc)
        combined = combined_signal(rule, sem, key)
        points = combined * 4.0  # each must-have max 4
        per_category[cat] = {
            "rule": rule, "semantic": round(sem, 3), "keyword": round(key, 3), "points": round(points, 3)
        }
        raw_total += points

    # score good-to-have categories (each up to 2 points)
    for cat in good_have:
        kws = KEYWORDS.get(cat, [])
        rule = 1.0 if any(kw in t for kw in kws) else 0.0
        key = keyword_score_single_category(t, kws)
        desc = DESCRIPTIONS.get(cat, DESCRIPTIONS["content_structure"])
        sem = semantic_similarity(transcript, desc)
        combined = combined_signal(rule, sem, key)
        points = combined * 2.0  # each good-to-have max 2
        per_category[cat] = {
            "rule": rule, "semantic": round(sem, 3), "keyword": round(key, 3), "points": round(points, 3)
        }
        raw_total += points

    # raw_total expected range: 0 - 30
    scaled_to_40 = (raw_total / 30.0) * WEIGHTS["content_structure"]
    return {"per_category": per_category, "raw_total": round(raw_total, 3), "scaled_score": round(scaled_to_40, 2)}

# --- Final pipeline assembly ---
def score_transcript(transcript: str, duration_sec: float = None):
    result = {}
    t = clean_text(transcript)

    # 1) Content & Structure (detailed)
    content_res = compute_content_structure_scores(transcript)
    result["content_structure"] = content_res

    # 2) Speech Rate (needs duration)
    if duration_sec is None:
        # if duration not provided, attempt a default or mark as None
        speech_points = None
        wpm = None
    else:
        wpm = compute_wpm(transcript, duration_sec)
        speech_points = speech_rate_score_from_wpm(wpm)
    result["speech_rate"] = {"wpm": wpm, "points": speech_points}

    # 3) Language & Grammar (20 points = grammar(10) + vocab(10))
    grammar_p = grammar_score_points(transcript)
    vocab_p = ttr_points(transcript)
    # sum to 20
    lang_points = grammar_p + vocab_p
    result["language_grammar"] = {"grammar_points": grammar_p, "vocab_points": vocab_p, "total": round(lang_points, 2)}

    # 4) Clarity (filler)
    clarity_p = clarity_points(transcript)  # already scaled to WEIGHTS["clarity"]
    result["clarity"] = {"filler_percent": round(filler_rate(transcript), 2), "points": round(clarity_p, 2)}

    # 5) Engagement (sentiment)
    engagement_p = engagement_points(transcript)
    result["engagement"] = {"points": round(engagement_p, 2)}

    # Sum up final total: content(scaled) + speech + language + clarity + engagement
    total = 0.0
    total += content_res["scaled_score"]  # out of 40
    total += (speech_points if speech_points is not None else 0.0)  # out of 10
    total += (lang_points if lang_points is not None else 0.0)  # out of 20
    total += clarity_p  # out of 15
    total += engagement_p  # out of 15

    result["overall_score"] = round(total, 2)
    return result

# --- Example run using provided sample transcript file ---
if __name__ == "__main__":
    SAMPLE_PATH = "./Sample text for case study.txt"  # your uploaded sample transcript
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample_text = f.read()

    # For the sample we know duration ~52s (from earlier notes); if unknown provide duration_sec argument
    out = score_transcript(sample_text, duration_sec=52.0)
    import json
    print(json.dumps(out, indent=2))
