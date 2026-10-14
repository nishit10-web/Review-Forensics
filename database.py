import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reviews.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'General',
            reviewer_name TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            review_text TEXT NOT NULL,
            is_verified INTEGER NOT NULL DEFAULT 0,
            review_category TEXT NOT NULL,
            product_verdict TEXT NOT NULL,
            product_quality_score REAL NOT NULL,
            astar_metrics_json TEXT NOT NULL,
            breakdown_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM reviews')
    if cursor.fetchone()[0] == 0:
        seed_sample_reviews(cursor)
        conn.commit()

    conn.close()

def seed_sample_reviews(cursor):
    sample_records = [
        (
            "SoundWave ANC Wireless Headphones",
            "Electronics",
            "David Miller",
            5,
            "Used these on a cross-country flight. Active noise cancellation is solid for this price point, battery lasted 28 hours, and ear cushions are plush. Minor con: companion app requires unnecessary login, but overall very satisfied.",
            1,
            "Genuine (Truthful) Review",
            "Good Product",
            92.0,
            json.dumps({"model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)", "confidence_score": 93.4, "predicted_category": "Genuine (Truthful) Review"}),
            json.dumps({"positive_sentiment": 0.8, "negative_sentiment": 0.1, "promotional_score": 0.0, "defamatory_score": 0.0, "style_anomaly": 0.05, "entropy_penalty": 0.05, "trust_factor": 0.85}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        (
            "MiracleGlow Vitamin C Face Cream",
            "Beauty & Skincare",
            "PromoQueen_2026",
            5,
            "AMAZING PRODUCT!!!! BEST EVER IN THE WORLD! BUY NOW GUYS GUARANTEED 100% SATISFACTION MIRACLE GLOWING SKIN! USE DISCOUNT CODE SAVE20 BUY TODAY FAST!!",
            0,
            "Promotional Spam (Paid / Incentivized)",
            "Artificially Overhyped (Biased)",
            50.0,
            json.dumps({"model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)", "confidence_score": 94.1, "predicted_category": "Promotional Spam (Paid / Incentivized)"}),
            json.dumps({"positive_sentiment": 0.95, "negative_sentiment": 0.0, "promotional_score": 0.95, "defamatory_score": 0.0, "style_anomaly": 0.85, "entropy_penalty": 0.20, "trust_factor": 0.10}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        (
            "ProChef Vortex Commercial Blender",
            "Home & Kitchen",
            "Anonymous_Buyer",
            1,
            "BUY FROM BLENDTEC INSTEAD! TOTAL SCAM COMPANY STEALING CUSTOMER MONEY WITH CRIMINAL DEFECTS! COMPLETE FRAUD BOYCOTT THIS PRODUCT IMMEDIATELY!! WORST TRASH EVER MADE!!",
            0,
            "Defamatory Spam (Competitor Attack)",
            "Unfairly Defamed (Smear Attack)",
            60.0,
            json.dumps({"model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)", "confidence_score": 95.6, "predicted_category": "Defamatory Spam (Competitor Attack)"}),
            json.dumps({"positive_sentiment": 0.0, "negative_sentiment": 0.95, "promotional_score": 0.0, "defamatory_score": 0.95, "style_anomaly": 0.90, "entropy_penalty": 0.25, "trust_factor": 0.05}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        (
            "UltraGrip Magnetic Car Mount",
            "Automotive",
            "bot_user_9921",
            5,
            "Good good good good good good good good good good good good good good good good.",
            0,
            "Bot / Non-Review Junk",
            "Indeterminate (Junk Discarded)",
            50.0,
            json.dumps({"model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)", "confidence_score": 88.9, "predicted_category": "Bot / Non-Review Junk"}),
            json.dumps({"positive_sentiment": 0.2, "negative_sentiment": 0.0, "promotional_score": 0.0, "defamatory_score": 0.0, "style_anomaly": 0.20, "entropy_penalty": 0.95, "trust_factor": 0.05}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        (
            "BudgetFit Smart Step Counter",
            "Wearables",
            "Elena Rostova",
            1,
            "The screen completely died after 48 hours. Plastic buckle snapped on day three while jogging. Pedometer was overcounting steps by nearly 40%. Cannot recommend, requested refund immediately.",
            1,
            "Genuine (Truthful) Review",
            "Bad Product",
            15.0,
            json.dumps({"model_name": "Supervised Logistic Regression (TF-IDF + Signal Fusion)", "confidence_score": 92.2, "predicted_category": "Genuine (Truthful) Review"}),
            json.dumps({"positive_sentiment": 0.0, "negative_sentiment": 0.85, "promotional_score": 0.0, "defamatory_score": 0.0, "style_anomaly": 0.10, "entropy_penalty": 0.05, "trust_factor": 0.85}),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    ]

    cursor.executemany('''
        INSERT INTO reviews (
            product_name, category, reviewer_name, rating, review_text,
            is_verified, review_category, product_verdict, product_quality_score,
            astar_metrics_json, breakdown_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_records)

def add_review(product_name, category, reviewer_name, rating, review_text, is_verified,
               review_category, product_verdict, product_quality_score, astar_metrics, breakdown):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reviews (
            product_name, category, reviewer_name, rating, review_text,
            is_verified, review_category, product_verdict, product_quality_score,
            astar_metrics_json, breakdown_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        product_name,
        category or 'General',
        reviewer_name,
        int(rating),
        review_text,
        1 if is_verified else 0,
        review_category,
        product_verdict,
        float(product_quality_score),
        json.dumps(astar_metrics) if isinstance(astar_metrics, dict) else str(astar_metrics),
        json.dumps(breakdown) if isinstance(breakdown, dict) else str(breakdown),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_reviews(category=None, search=None, limit=50, offset=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM reviews WHERE 1=1"
    params = []

    if category and category.lower() != 'all':
        query += " AND review_category = ?"
        params.append(category)

    if search:
        query += " AND (product_name LIKE ? OR reviewer_name LIKE ? OR review_text LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "product_name": r["product_name"],
            "category": r["category"],
            "reviewer_name": r["reviewer_name"],
            "rating": r["rating"],
            "review_text": r["review_text"],
            "is_verified": bool(r["is_verified"]),
            "review_category": r["review_category"],
            "product_verdict": r["product_verdict"],
            "product_quality_score": r["product_quality_score"],
            "ml_metrics": json.loads(r["astar_metrics_json"]),
            "astar_metrics": json.loads(r["astar_metrics_json"]),
            "breakdown": json.loads(r["breakdown_json"]),
            "created_at": r["created_at"]
        })
    conn.close()
    return results

def delete_review(review_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_reviews():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reviews")
    conn.commit()
    conn.close()

def get_review_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE review_category = 'Genuine (Truthful) Review'")
    genuine_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE review_category = 'Promotional Spam (Paid / Incentivized)'")
    promo_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE review_category = 'Defamatory Spam (Competitor Attack)'")
    defamy_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reviews WHERE review_category = 'Bot / Non-Review Junk'")
    bot_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(rating) FROM reviews")
    avg_rating_row = cursor.fetchone()[0]
    avg_rating = round(avg_rating_row, 2) if avg_rating_row else 0.0

    cursor.execute("SELECT AVG(product_quality_score) FROM reviews WHERE review_category = 'Genuine (Truthful) Review'")
    avg_qual_row = cursor.fetchone()[0]
    avg_genuine_quality = round(avg_qual_row, 1) if avg_qual_row else 0.0

    conn.close()
    return {
        "total": total,
        "genuine": genuine_count,
        "promotional": promo_count,
        "defamatory": defamy_count,
        "bot_junk": bot_count,
        "avg_nominal_rating": avg_rating,
        "avg_genuine_quality": avg_genuine_quality,
        "spam_percentage": round(((promo_count + defamy_count + bot_count) / max(1, total)) * 100, 1)
    }
