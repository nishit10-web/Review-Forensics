import re
import math
from collections import Counter

# 1. Promotional Spam Keywords (Overhyped / Paid / Incentivized)
PROMOTIONAL_PATTERNS = [
    r"\b100% (satisfied|guaranteed|works|recommend|real)?\b",
    r"\b100%\b",
    r"\bmiracle\b",
    r"\bbest (product|ever|in the world|on the market|purchase)\b",
    r"\bbuy (now|immediately|today|fast)\b",
    r"\bmust buy\b",
    r"\bamazing product\b",
    r"\bfree (sample|gift|replacement|item received)\b",
    r"\bpaid (review|promotion|to test)\b",
    r"\bdiscount (code|coupon|link)\b",
    r"\bguaranteed satisfaction\b",
    r"\bguaranteed\b",
    r"\bact now\b",
    r"\blife changing\b",
    r"\bmagic product\b",
    r"\bclick (here|the link)\b",
    r"\bgive this 10 stars\b",
    r"\brated 10/10\b",
    r"\bunbelievable results\b",
    r"\bhighest recommendation\b"
]

# 2. Defamatory Spam Keywords (Competitor Attacks / Malicious Smear)
DEFAMATORY_PATTERNS = [
    r"\bbuy (an? )?(another|competitor|other|better) (brand|product|one) instead\b",
    r"\bbuy from .* instead\b",
    r"\b(total|absolute) scam\b",
    r"\bscam company\b",
    r"\bstealing (our|my|your) money\b",
    r"\bfake sellers?\b",
    r"\bboycott this\b",
    r"\bcriminal company\b",
    r"\bcomplete fraud\b",
    r"\bworst trash (ever|made)\b",
    r"\bdo not trust this seller\b",
    r"\billegal product\b",
    r"\bpoison\b",
    r"\bgive zero stars\b",
    r"\brated 0/10\b",
    r"\breport to fbi|report to authorities\b",
    r"\bcheap chinese (knockoff|junk|garbage)\b"
]

# 3. AI / LLM Synthetic Bot Review Patterns (ChatGPT, Claude, Gemini markers)
AI_LLM_PATTERNS = [
    r"\bin conclusion\b",
    r"\ball in all\b",
    r"\bto sum (it )?up\b",
    r"\bin summary\b",
    r"\boverall,? i (highly|strongly|definitely) recommend\b",
    r"\boverall,? (it is|the product is|this is) a (testament|must-have|game-changer)\b",
    r"\btestament to\b",
    r"\bgame[- ]changer\b",
    r"\bstands out as\b",
    r"\bdelv(e|es|ing) (into|deep)\b",
    r"\ba must[- ]have (addition|item|for|in your)\b",
    r"\bwhether you('re| are) a .* or a .*\b",
    r"\bboasts an? (impressive|array|exceptional|sleek|powerful)\b",
    r"\bsleek and (stylish|modern|elegant|compact)\b",
    r"\bseamless (integration|experience|connectivity|transition|blend)\b",
    r"\braises the bar\b",
    r"\bupon unboxing\b",
    r"\bin terms of (performance|quality|design|battery|durability)\b",
    r"\bexceptional blend of\b",
    r"\bexceeded (all )?my expectations\b",
    r"\blook no further than\b",
    r"\bwithout a shadow of a doubt\b",
    r"\btop-notch (performance|build|quality|experience)\b",
    r"\bimpressive piece of (tech|hardware|engineering|machinery|craftsmanship)\b",
    r"\btick(s)? all the (right )?boxes\b",
    r"\bhits the sweet spot\b",
    r"\bmeticulously (designed|crafted|engineered)\b",
    r"\bpleasantly surprised by the\b",
    r"\bcannot recommend (it )?enough\b",
    r"\bfurthermore,? \b",
    r"\bmoreover,? \b",
    r"\bpros and cons\b",
    r"\bnot without its flaws\b",
    r"\ball things considered\b",
    r"\bfirst and foremost\b",
    r"\bsets a new standard\b",
    r"\bexceptional build quality\b",
    r"\bstellar (battery life|performance|display|audio)\b",
    r"\ba force to be reckoned with\b",
    r"\bpushing the boundaries of\b",
    r"\ba worthy investment\b",
    r"\bit'?s safe to say\b",
    r"\bdelivers on all fronts\b",
    r"\bworth every penny\b"
]

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "perfect",
    "love", "best", "superb", "brilliant", "exceptional", "outstanding", "flawless",
    "impressive", "spectacular", "incredible", "satisfied", "happy", "recommend",
    "durable", "reliable", "comfortable", "smooth", "high-quality", "efficient"
}

NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "bad", "worst", "poor", "broken", "useless",
    "waste", "scam", "defective", "disappointed", "hate", "junk", "fake", "garbage",
    "regret", "damaged", "trash", "fraud", "clunky", "noisy", "failed", "flimsy"
}

def extract_features(rating, review_text, is_verified, reviewer_name=""):
    """
    Extracts a normalized 8-dimensional feature vector in [0.0, 1.0] for A* search:
    [0] norm_rating: Rating / 5.0
    [1] pos_sentiment: Positive sentiment score
    [2] neg_sentiment: Negative sentiment score
    [3] promo_score: Promotional / Astroturfing keyword density
    [4] defamy_score: Defamatory / Competitor smear keyword density
    [5] style_anomaly: ALL CAPS and exclamation cluster penalty
    [6] entropy_penalty: Low vocabulary diversity / bot repetition penalty
    [7] trust_factor: Purchase verification and reviewer alias credibility
    """
    text = (review_text or "").strip()
    text_lower = text.lower()
    rating = int(rating)
    is_verified = bool(is_verified)
    reviewer = (reviewer_name or "").strip().lower()

    # Words and length
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text_lower)
    word_count = len(words)

    # 1. Normalized Rating
    norm_rating = max(0.0, min(1.0, rating / 5.0))

    # 2 & 3. Sentiment Scores
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    pos_sentiment = min(1.0, pos_count / max(3, word_count * 0.35))
    neg_sentiment = min(1.0, neg_count / max(3, word_count * 0.35))

    # 4. Promotional Buzzwords
    promo_matches = []
    for pat in PROMOTIONAL_PATTERNS:
        found = re.findall(pat, text_lower)
        if found:
            clean = pat.replace(r"\b", "").replace("(", "").replace(")", "").replace("|", "/")
            promo_matches.append(clean)
    promo_score = min(1.0, len(promo_matches) * 0.35)

    # 5. Defamatory / Competitor Attack Buzzwords
    defamy_matches = []
    for pat in DEFAMATORY_PATTERNS:
        found = re.findall(pat, text_lower)
        if found:
            clean = pat.replace(r"\b", "").replace("(", "").replace(")", "").replace("|", "/")
            defamy_matches.append(clean)
    defamy_score = min(1.0, len(defamy_matches) * 0.40)

    # 6. Style Anomaly (ALL CAPS, Exclamation clusters)
    style_anomaly = 0.0
    if len(text) > 0:
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / len(text)
        exclamation_count = text.count('!')

        if caps_ratio > 0.40 and len(text) > 15:
            style_anomaly += 0.50
        elif caps_ratio > 0.25:
            style_anomaly += 0.25

        if exclamation_count >= 4:
            style_anomaly += min(0.50, exclamation_count * 0.10)
        elif exclamation_count >= 2:
            style_anomaly += 0.15

        # Repeated character stretches like "goooood"
        if re.search(r'(.)\1{3,}', text):
            style_anomaly += 0.20
    style_anomaly = min(1.0, style_anomaly)

    # 7. Low Entropy / Repetition / Bot Junk & AI Detection
    entropy_penalty = 0.0
    if word_count < 4:
        entropy_penalty = 0.85
    elif word_count > 0:
        unique_words = set(words)
        vocab_ratio = len(unique_words) / word_count
        if vocab_ratio < 0.40 and word_count >= 6:
            entropy_penalty = 0.95
        elif vocab_ratio < 0.60:
            entropy_penalty = 0.50
        
        # Check repeated identical tokens
        counts = Counter(words)
        most_common_freq = counts.most_common(1)[0][1] if counts else 0
        if most_common_freq >= word_count * 0.45 and word_count >= 5:
            entropy_penalty = max(entropy_penalty, 0.90)

    # AI / LLM Synthetic Text Detection (ChatGPT / Gemini / Claude)
    ai_matches = []
    for pat in AI_LLM_PATTERNS:
        found = re.findall(pat, text_lower)
        if found:
            clean = pat.replace(r"\b", "").replace("(", "").replace(")", "").replace("|", "/")
            ai_matches.append(clean)
    ai_matches = list(set(ai_matches))
    ai_match_count = len(ai_matches)

    if ai_match_count >= 3:
        ai_score = 1.0
    elif ai_match_count == 2:
        ai_score = 0.80
    elif ai_match_count == 1:
        ai_score = 0.50 if word_count > 25 else 0.25
    else:
        ai_score = 0.0

    # Composite Bot / Synthetic Anomaly
    bot_anomaly = min(1.0, max(entropy_penalty, ai_score))

    # 8. Trust Factor
    trust_factor = 0.20 # baseline
    if is_verified:
        trust_factor += 0.60
    if re.match(r'^(user[_\-\d]+|customer|shopper|anonymous|bot\d*)$', reviewer):
        trust_factor = max(0.0, trust_factor - 0.35)
    if ai_score >= 0.50:
        trust_factor = 0.0 # AI generated has zero trust

    return {
        "vector": [
            norm_rating,
            round(pos_sentiment, 3),
            round(neg_sentiment, 3),
            round(promo_score, 3),
            round(defamy_score, 3),
            round(style_anomaly, 3),
            round(bot_anomaly, 3),
            round(trust_factor, 3),
            round(ai_score, 3)
        ],
        "word_count": word_count,
        "promo_matches": list(set(promo_matches)),
        "defamy_matches": list(set(defamy_matches)),
        "ai_matches": ai_matches,
        "ai_score": round(ai_score, 3),
        "is_ai_generated": bool(ai_score >= 0.50),
        "is_verified": is_verified,
        "rating": rating
    }
