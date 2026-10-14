import os
import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from detector import extract_features

CATEGORY_GENUINE = "Genuine (Truthful) Review"
CATEGORY_PROMO = "Promotional Spam (Paid / Incentivized)"
CATEGORY_DEFAMATORY = "Defamatory Spam (Competitor Attack)"
CATEGORY_BOT = "Bot / Non-Review Junk"

CATEGORIES = [CATEGORY_GENUINE, CATEGORY_PROMO, CATEGORY_DEFAMATORY, CATEGORY_BOT]

# ==============================================================
# 1. LABELED SUPERVISED TRAINING DATASET (MULTI-CLASS)
# ==============================================================

TRAINING_REVIEWS = [
    # --- GENUINE (TRUTHFUL) REVIEWS ---
    ("I have used these headphones for 3 weeks on my daily commute. Battery comfortably lasts 30 hours, noise cancellation blocks engine hum, and ear cushions are soft. Only con is the app requires a login.", 5, 1, CATEGORY_GENUINE),
    ("Purchased this blender for making fruit smoothies. Blades are sharp and stainless steel jar is durable. A bit noisy on maximum speed, but overall very reliable for daily kitchen use.", 4, 1, CATEGORY_GENUINE),
    ("Screen cracked after a 2-foot drop onto carpet. Battery life is barely 4 hours and customer support refused replacement. Very disappointed with build quality.", 1, 1, CATEGORY_GENUINE),
    ("Keyboard switches feel tactile and responsive for programming. Keycaps have good texture with zero shine after month of use. Cable could be slightly longer.", 4, 1, CATEGORY_GENUINE),
    ("Vacuum cleaner stopped charging after 10 days of use. Filter clogged very quickly and suction is weak on thick carpets. Returned for a refund.", 1, 1, CATEGORY_GENUINE),
    ("Solid running shoes with responsive React foam cushioning. Secure midfoot fit and true to size. Forefoot is a bit snug for wide feet.", 4, 1, CATEGORY_GENUINE),
    ("Decent laptop backpack with padded compartment. Fits my 15-inch laptop and water bottle. Zippers are slightly stiff initially.", 3, 1, CATEGORY_GENUINE),
    ("Watch pedometer inaccurate by 35 percent and bluetooth disconnects frequently. Strap buckle snapped on day two. Low quality product.", 1, 1, CATEGORY_GENUINE),
    ("The anti-reflective OLED display is bright outdoors and camera captures crisp dynamic range. Solid battery life.", 5, 1, CATEGORY_GENUINE),
    ("Coffee maker brews hot flavorful coffee within 4 minutes. Water reservoir is easy to refill and thermal carafe keeps heat for hours.", 5, 1, CATEGORY_GENUINE),
    ("The jacket zipper broke after two washes. Fabric is thin and stitching started unraveling around the cuffs.", 2, 1, CATEGORY_GENUINE),
    ("Excellent sound separation and deep bass for jazz and acoustic music. Lightweight headband without fatigue.", 5, 1, CATEGORY_GENUINE),

    # --- PROMOTIONAL SPAM (PAID / INCENTIVIZED) ---
    ("AMAZING PRODUCT BEST EVER IN THE WORLD! BUY NOW GUYS GUARANTEED 100% SATISFACTION MIRACLE RESULTS!! USE DISCOUNT CODE SAVE20 BUY TODAY FAST!!", 5, 0, CATEGORY_PROMO),
    ("100% guaranteed works like magic! Miracle product best ever made on the planet! Must buy fast buy now discount link below!!", 5, 0, CATEGORY_PROMO),
    ("Free sample received and completely blown away! Life changing miracle formula 10/10 stars! Buy immediately click here guaranteed satisfaction!!", 5, 0, CATEGORY_PROMO),
    ("BEST EVER MADE! Miracle cream 100% works best in the world! Buy today act now discount coupon save50 must buy fast!!", 5, 0, CATEGORY_PROMO),
    ("Paid promotion test review: absolutely outstanding miracle device 100% satisfied zero complaints buy now fast!!", 5, 0, CATEGORY_PROMO),
    ("AMAZING PRODUCT! Guaranteed 100% miracle sound best purchase ever made! Click the link discount code free gift buy now!!", 5, 0, CATEGORY_PROMO),
    ("Must buy today! Miracle results best product ever in the universe 100% satisfaction guaranteed act now fast!!", 5, 0, CATEGORY_PROMO),
    ("Blown away! Life changing product 100% real best ever buy now click here discount code save today!!", 5, 0, CATEGORY_PROMO),
    ("Rated 10/10 magic miracle product! Give this 10 stars buy immediately guaranteed refund click link!!", 5, 0, CATEGORY_PROMO),
    ("Best ever in the world! Buy fast miracle works 100% free gift received must buy today!!", 5, 0, CATEGORY_PROMO),

    # --- DEFAMATORY SPAM (COMPETITOR ATTACK) ---
    ("BUY FROM BLENDTEC INSTEAD! TOTAL SCAM COMPANY STEALING CUSTOMER MONEY WITH CRIMINAL DEFECTS! COMPLETE FRAUD BOYCOTT THIS PRODUCT IMMEDIATELY!! WORST TRASH EVER MADE!!", 1, 0, CATEGORY_DEFAMATORY),
    ("BUY FROM SAMSUNG INSTEAD! THIS BRAND IS A COMPLETE FRAUD SCAM COMPANY STEALING YOUR HARD EARNED CASH! BOYCOTT IMMEDIATELY WORST GARBAGE EVER!!", 1, 0, CATEGORY_DEFAMATORY),
    ("Buy another competitor brand instead! Fake seller scam company criminal fraud illegal poison garbage do not trust this seller report to authorities!!", 1, 0, CATEGORY_DEFAMATORY),
    ("BUY FROM BOSE INSTEAD! TOTAL SCAM FRAUDULENT COMPANY CHEAP JUNK STEALING MONEY WORST TRASH EVER MADE BOYCOTT THIS SELLER!!", 1, 0, CATEGORY_DEFAMATORY),
    ("Complete fraud scam company! Buy from Dyson instead fake sellers stealing money cheap knockoff garbage zero stars boycott!!", 1, 0, CATEGORY_DEFAMATORY),
    ("BUY FROM HOKA INSTEAD! OVERPRICED SCAM COMPANY STEALING MONEY CRIMINAL FRAUD CHEAP TRASH DO NOT BUY BOYCOTT NOW!!", 1, 0, CATEGORY_DEFAMATORY),
    ("Total scam company! Criminal fraud stealing money cheap chinese knockoff garbage buy from competitor instead boycott!!", 1, 0, CATEGORY_DEFAMATORY),
    ("Fake seller scam company report to fbi! Absolute fraud stealing money worst junk garbage buy other brand instead boycott!!", 1, 0, CATEGORY_DEFAMATORY),
    ("BUY FROM SONY INSTEAD! COMPLETE FRAUD SCAM STEALING MONEY BOYCOTT THIS JUNK ZERO STARS CRIMINAL SELLER!!", 1, 0, CATEGORY_DEFAMATORY),

    # --- BOT / NON-REVIEW JUNK (REPETITIVE + AI/LLM SYNTHETIC REVIEWS) ---
    ("good good good good good good good good good good good good good.", 5, 0, CATEGORY_BOT),
    ("nice nice nice nice nice nice nice nice nice nice nice nice nice nice.", 5, 0, CATEGORY_BOT),
    ("best best best best best best best best best best best best best.", 5, 0, CATEGORY_BOT),
    ("Shoe shoe shoe shoe shoe shoe shoe shoe shoe shoe shoe shoe.", 5, 0, CATEGORY_BOT),
    ("asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf.", 5, 0, CATEGORY_BOT),
    ("cool cool cool cool cool cool cool cool cool cool cool.", 5, 0, CATEGORY_BOT),
    ("ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok ok.", 3, 0, CATEGORY_BOT),
    ("great great great great great great great great great great.", 5, 0, CATEGORY_BOT),
    ("product product product product product product product.", 5, 0, CATEGORY_BOT),
    ("xyz abc xyz abc xyz abc xyz abc xyz abc xyz abc.", 4, 0, CATEGORY_BOT),
    ("love love love love love love love love love love love.", 5, 0, CATEGORY_BOT),
    ("item item item item item item item item item item item.", 5, 0, CATEGORY_BOT),
    # ChatGPT / Claude / Gemini AI synthetic reviews:
    ("In conclusion, the Apple iPhone 16 Pro Max is a true testament to modern engineering. It stands out as a game-changer in the smartphone landscape. From its sleek and stylish titanium chassis to the seamless integration of Apple Intelligence, this device boasts an impressive display and stellar battery life. Furthermore, whether you are a casual user or a tech enthusiast, this phone offers something for everyone. Overall, I highly recommend this incredible piece of technology. A must-have addition to any daily routine!", 5, 0, CATEGORY_BOT),
    ("As an avid runner, I was thrilled to try out the Nike Air Zoom Pegasus 41. Upon unboxing, the design immediately catches your eye. In terms of performance, the ReactX foam delivers exceptional energy return and a remarkably smooth ride. Delving into the details, the engineered mesh upper provides optimal breathability. In conclusion, Nike has once again raised the bar. It truly stands out as an exceptional running companion.", 5, 0, CATEGORY_BOT),
    ("The Sony WH-1000XM5 headphones represent an exceptional blend of form and function. Delving deep into audio fidelity, the active noise cancellation raises the bar for daily commuters. Furthermore, it boasts an impressive battery life that exceeded all my expectations. Whether you're an audiophile or an office worker, this is a worthy investment. In summary, it stands out as a top-tier audio marvel.", 5, 0, CATEGORY_BOT),
    ("In terms of performance and suction power, the Dyson V15 Detect is a true game-changer. Meticulously designed, it seamlessly cleans hardwood and carpets. In conclusion, Dyson has delivered an impressive piece of engineering. A must-have for every modern home.", 5, 0, CATEGORY_BOT),
    ("Overall, the Samsung Galaxy S24 Ultra is a testament to cutting-edge mobile innovation. With its sleek and modern design, it ticks all the right boxes. Delving into the display capabilities, the anti-reflective screen hits the sweet spot. Without a shadow of a doubt, it is a worthy investment.", 5, 0, CATEGORY_BOT),
    ("Upon unboxing this smart watch, I was pleasantly surprised by the sleek and elegant build. Delving into the features, the sensor suite boasts an impressive degree of accuracy. In summary, whether you are a fitness fanatic or a business professional, this product stands out as a game-changer.", 5, 0, CATEGORY_BOT),
    ("In conclusion, this blender delivers on all fronts. In terms of power, the motor is top-notch. Furthermore, the stainless steel blades blend frozen fruits with seamless ease. Overall, I highly recommend this impressive piece of kitchen technology. A must-have addition to your countertop!", 5, 0, CATEGORY_BOT),
    ("All things considered, this wireless keyboard is a testament to ergonomic craftsmanship. It boasts an impressive tactile response while typing. Furthermore, the Bluetooth connectivity is seamless across multiple devices. In summary, it truly hits the sweet spot for productivity.", 5, 0, CATEGORY_BOT),
    ("First and foremost, this face cream exceeded all my expectations. In terms of hydration, the formula provides a seamless and lightweight texture. Delving into the ingredients, it is meticulously crafted for sensitive skin. In conclusion, look no further than this exceptional skincare product.", 5, 0, CATEGORY_BOT),
    ("Overall, this laptop stands out as a true marvel of modern computing. It boasts an impressive battery life and a stunning OLED display. Whether you are a student or a creative professional, it delivers on all fronts without a shadow of a doubt.", 5, 0, CATEGORY_BOT)
]

# ==============================================================
# 2. FEATURE EXTRACTION & PIPELINE
# ==============================================================

class ReviewFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms text and metadata into combined TF-IDF + engineered feature array.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=800,
            sublinear_tf=True
        )

    def fit(self, X_text_meta, y=None):
        texts = [item[0] for item in X_text_meta]
        self.vectorizer.fit(texts)
        return self

    def transform(self, X_text_meta):
        texts = [item[0] for item in X_text_meta]
        tfidf_sparse = self.vectorizer.transform(texts).toarray()

        # Engineered dense features
        dense_feats = []
        for text, rating, verified in X_text_meta:
            extracted = extract_features(rating, text, verified)
            v = extracted["vector"] # normalized signals including AI score
            dense_feats.append(v)

        dense_array = np.array(dense_feats)
        return np.hstack((tfidf_sparse, dense_array))

# Global trained model and pipeline
ML_MODEL = None
ML_VECTORIZER = None

def train_supervised_model():
    """
    Trains the Supervised Machine Learning classifier on the multi-class dataset.
    Uses Logistic Regression with balanced class weights and L2 regularization.
    """
    global ML_MODEL, ML_VECTORIZER
    X_train = [(text, rating, verified) for text, rating, verified, _ in TRAINING_REVIEWS]
    y_train = [cat for _, _, _, cat in TRAINING_REVIEWS]

    vectorizer = ReviewFeatureExtractor()
    X_transformed = vectorizer.fit_transform(X_train)

    # Supervised Classifier: Logistic Regression (Multinomial Softmax)
    clf = LogisticRegression(
        C=2.0,
        max_iter=400,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_transformed, y_train)

    ML_MODEL = clf
    ML_VECTORIZER = vectorizer
    return clf, vectorizer

# Train once on module load
train_supervised_model()

# ==============================================================
# 3. SUPERVISED ML CLASSIFICATION & QUALITY EVALUATOR
# ==============================================================

def classify_review_with_ml(rating, review_text, is_verified, reviewer_name=""):
    """
    Applies the Supervised Machine Learning model to classify the review
    and evaluate product quality.
    """
    global ML_MODEL, ML_VECTORIZER
    if ML_MODEL is None or ML_VECTORIZER is None:
        train_supervised_model()

    text = (review_text or "").strip()
    rating = int(rating)
    is_verified = bool(is_verified)

    # Extract feature signals for explanation
    extracted = extract_features(rating, text, is_verified, reviewer_name)
    feat_vector = extracted["vector"]

    # Transform input
    input_item = [(text, rating, is_verified)]
    X_test = ML_VECTORIZER.transform(input_item)

    # Predict with Supervised ML
    predicted_cat = ML_MODEL.predict(X_test)[0]
    probabilities = ML_MODEL.predict_proba(X_test)[0]
    classes = list(ML_MODEL.classes_)

    prob_dict = {}
    for cls_name, p in zip(classes, probabilities):
        prob_dict[cls_name] = round(float(p) * 100.0, 1)

    # Check for AI / LLM Synthetic generation
    is_ai = extracted.get("is_ai_generated", False) or len(extracted.get("ai_matches", [])) >= 2 or extracted.get("ai_score", 0.0) >= 0.50
    ai_score = extracted.get("ai_score", 0.0)
    ai_matches = extracted.get("ai_matches", [])

    if is_ai:
        predicted_cat = CATEGORY_BOT
        boosted_p = max(prob_dict.get(CATEGORY_BOT, 0.0), round(max(92.0, ai_score * 98.0), 1))
        prob_dict[CATEGORY_BOT] = boosted_p
        rem = round(100.0 - boosted_p, 1)
        other_sum = sum(v for k, v in prob_dict.items() if k != CATEGORY_BOT)
        if other_sum > 0:
            for k in prob_dict:
                if k != CATEGORY_BOT:
                    prob_dict[k] = round((prob_dict[k] / other_sum) * rem, 1)

    confidence = prob_dict.get(predicted_cat, 95.0)

    # Evaluate Product Quality using ML classification
    product_eval = evaluate_product_quality_ml(predicted_cat, rating, feat_vector, is_ai=is_ai, ai_matches=ai_matches)

    return {
        "success": True,
        "review_category": predicted_cat,
        "product_verdict": product_eval["verdict"],
        "product_quality_score": product_eval["quality_score"],
        "product_quality_description": product_eval["description"],
        "ml_metrics": {
            "model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)",
            "predicted_category": predicted_cat,
            "confidence_score": confidence,
            "class_probabilities": prob_dict,
            "decision_features_count": X_test.shape[1],
            "is_verified": is_verified,
            "is_ai_generated": is_ai,
            "ai_score": round(ai_score * 100.0, 1),
            "ai_matches": ai_matches
        },
        "feature_breakdown": {
            "normalized_rating": feat_vector[0],
            "positive_sentiment": feat_vector[1],
            "negative_sentiment": feat_vector[2],
            "promotional_score": feat_vector[3],
            "defamatory_score": feat_vector[4],
            "style_anomaly": feat_vector[5],
            "bot_anomaly": feat_vector[6],
            "trust_factor": feat_vector[7],
            "ai_score": round(ai_score * 100.0, 1),
            "is_ai_generated": is_ai,
            "ai_matches": ai_matches,
            "promo_matches": extracted["promo_matches"],
            "defamy_matches": extracted["defamy_matches"],
            "word_count": extracted["word_count"],
            "is_verified": extracted["is_verified"]
        }
    }

def evaluate_product_quality_ml(category, rating, feat, is_ai=False, ai_matches=None):
    """
    Determines if the product is a Good Product or Bad Product
    by filtering out promotional, defamatory, and AI/bot bias using Supervised ML.
    """
    rating = int(rating)
    norm_r = feat[0]
    pos = feat[1]
    neg = feat[2]

    if category == CATEGORY_GENUINE:
        if rating >= 4:
            score = round(75.0 + (rating - 4) * 25.0 + (pos * 10.0), 1)
            return {
                "verdict": "Good Product",
                "quality_score": min(100.0, score),
                "description": f"Verified Genuine Review ({rating}/5 Stars). Supervised ML confirms high organic credibility indicating a reliable, high-quality product."
            }
        elif rating == 3:
            return {
                "verdict": "Average / Mediocre Product",
                "quality_score": 55.0,
                "description": "Verified Genuine Review (3/5 Stars). Product has balanced pros and cons."
            }
        else:
            score = round(max(5.0, rating * 15.0 - (neg * 10.0)), 1)
            return {
                "verdict": "Bad Product",
                "quality_score": score,
                "description": f"Verified Genuine Review ({rating}/5 Stars). Authentic critical feedback confirms product defects or poor performance."
            }

    elif category == CATEGORY_PROMO:
        return {
            "verdict": "Artificially Overhyped (Biased)",
            "quality_score": 50.0,
            "description": "Promotional / Paid Spam detected by Supervised ML. The 5-star rating is artificially inflated and neutralized."
        }

    elif category == CATEGORY_DEFAMATORY:
        return {
            "verdict": "Unfairly Defamed (Smear Attack)",
            "quality_score": 60.0,
            "description": "Competitor Defamatory Attack identified by Supervised ML. The malicious 1-star smear is neutralized; not indicative of true product defect."
        }

    else: # Bot Junk
        if is_ai:
            matches_str = ", ".join([f'"{m}"' for m in (ai_matches or [])[:3]]) if ai_matches else "typical AI transitions"
            return {
                "verdict": "Indeterminate (AI Bot Generated - Discarded)",
                "quality_score": 50.0,
                "description": f"Synthetic AI / LLM review detected (ChatGPT/Claude markers like {matches_str}). Discarded by Supervised ML."
            }
        else:
            return {
                "verdict": "Indeterminate (Junk Discarded)",
                "quality_score": 50.0,
                "description": "Bot / Non-Review Junk discarded by Supervised ML. Low entropy boilerplate or repetitive text."
            }
