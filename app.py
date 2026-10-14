from flask import Flask, render_template, request, jsonify, send_file
import json
import os

from database import init_db, add_review, get_reviews, delete_review, clear_all_reviews, get_review_stats
from ml_classifier import classify_review_with_ml, CATEGORIES
from brands_dataset import get_processed_brand_reviews, add_dynamic_brand_review
from sorting_algorithms import run_genetic_algorithm_sorting, run_local_search_sorting
from csp_solver import solve_top4_featured_csp, PRESET_CSP_SCENARIOS, check_sentiment_star_congruence, extract_aspects
from url_scraper import batch_audit_url_reviews

app = Flask(__name__)

# Initialize SQLite database on startup
init_db()

def resolve_brand(data):
    brand = (data.get('brand') or '').strip()
    if brand:
        return brand.title()
    pn = (data.get('product_name') or '').lower()
    cat = (data.get('category') or '').lower()
    for b in ['Apple', 'Nike', 'Sony', 'Samsung', 'Dyson', 'Adidas', 'Bose', 'Dell', 'Lenovo', 'Google', 'Asus', 'Puma']:
        if b.lower() in pn or b.lower() in cat:
            return b
    words = (data.get('product_name') or '').strip().split()
    if words:
        return words[0].title()
    return "General"

def compute_review_six_class_scores(result, rating):
    cat = result['review_category']
    is_ai = result['ml_metrics'].get('is_ai_generated', False)
    prob_dict = result['ml_metrics'].get('class_probabilities', {})
    
    gen_p = prob_dict.get("Genuine (Truthful) Review", 0.0)
    promo_p = prob_dict.get("Promotional Spam (Paid / Incentivized)", 0.0)
    defam_p = prob_dict.get("Defamatory Spam (Competitor Attack)", 0.0)
    bot_p = prob_dict.get("Bot / Non-Review Junk", 0.0)

    return {
        "genuine_good": round(gen_p if int(rating) >= 4 else (gen_p * 0.1), 1),
        "genuine_bad": round(gen_p if int(rating) < 4 else (gen_p * 0.1), 1),
        "promo_spam": promo_p,
        "defamatory": defam_p,
        "bot_repetitive": round(bot_p * (0.05 if is_ai else 1.0), 1),
        "ai_generated": round(bot_p * (1.0 if is_ai else 0.05), 1)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report')
@app.route('/docs')
def serve_report():
    return send_file('PROJECT_DOCUMENTATION_AND_VIVA_GUIDE.html')

# ==============================================================
# FAKE PRODUCT REVIEW DETECTOR & SUPERVISED ML REST APIS
# ==============================================================

@app.route('/api/reviews/scrape_url', methods=['POST'])
def api_scrape_url_reviews():
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    auto_save = bool(data.get('auto_save', True))

    if not url:
        return jsonify({'success': False, 'error': 'A valid website or product URL is required.'}), 400

    try:
        result = batch_audit_url_reviews(url, auto_save=auto_save)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to scrape and audit reviews: {str(e)}'}), 500

@app.route('/api/reviews/analyze', methods=['POST'])
def api_analyze_review():
    data = request.get_json() or {}
    product_name = data.get('product_name', 'General Product')
    category = data.get('category', 'Electronics')
    reviewer_name = data.get('reviewer_name', 'Anonymous')
    rating = int(data.get('rating', 5))
    review_text = data.get('review_text', '').strip()
    is_verified = bool(data.get('is_verified', False))
    brand = resolve_brand(data)

    if not review_text:
        return jsonify({'success': False, 'error': 'Review text is required.'}), 400

    result = classify_review_with_ml(
        rating=rating,
        review_text=review_text,
        is_verified=is_verified,
        reviewer_name=reviewer_name
    )

    result['ml_metrics']['six_class_scores'] = compute_review_six_class_scores(result, rating)

    return jsonify({
        'success': True,
        'product_name': product_name,
        'category': category,
        'brand': brand,
        'reviewer_name': reviewer_name,
        'rating': rating,
        'review_text': review_text,
        'is_verified': is_verified,
        'review_category': result['review_category'],
        'product_verdict': result['product_verdict'],
        'product_quality_score': result['product_quality_score'],
        'product_quality_description': result['product_quality_description'],
        'ml_metrics': result['ml_metrics'],
        'feature_breakdown': result['feature_breakdown']
    })

@app.route('/api/reviews', methods=['POST'])
def api_save_review():
    data = request.get_json() or {}
    product_name = data.get('product_name', 'General Product')
    category = data.get('category', 'Electronics')
    reviewer_name = data.get('reviewer_name', 'Anonymous')
    rating = int(data.get('rating', 5))
    review_text = data.get('review_text', '').strip()
    is_verified = bool(data.get('is_verified', False))
    brand = resolve_brand(data)

    if not review_text:
        return jsonify({'success': False, 'error': 'Review text is required.'}), 400

    result = classify_review_with_ml(
        rating=rating,
        review_text=review_text,
        is_verified=is_verified,
        reviewer_name=reviewer_name
    )

    result['ml_metrics']['six_class_scores'] = compute_review_six_class_scores(result, rating)

    review_id = add_review(
        product_name=product_name,
        category=category,
        reviewer_name=reviewer_name,
        rating=rating,
        review_text=review_text,
        is_verified=is_verified,
        review_category=result['review_category'],
        product_verdict=result['product_verdict'],
        product_quality_score=result['product_quality_score'],
        astar_metrics=result['ml_metrics'],
        breakdown=result['feature_breakdown']
    )

    # Auto-add to Brand Review tab
    brand_item = add_dynamic_brand_review(
        brand=brand,
        product_name=product_name,
        reviewer_name=reviewer_name,
        rating=rating,
        review_text=review_text,
        is_verified=is_verified,
        source="Added via Fake Review Detector"
    )

    return jsonify({
        'success': True,
        'id': review_id,
        'product_name': product_name,
        'category': category,
        'brand': brand,
        'added_to_brand_tab': True,
        'brand_item_id': brand_item['id'],
        'brand_message': f'Review also added to Brand review tab under "{brand}"',
        'reviewer_name': reviewer_name,
        'rating': rating,
        'review_text': review_text,
        'is_verified': is_verified,
        'review_category': result['review_category'],
        'product_verdict': result['product_verdict'],
        'product_quality_score': result['product_quality_score'],
        'product_quality_description': result['product_quality_description'],
        'ml_metrics': result['ml_metrics'],
        'feature_breakdown': result['feature_breakdown']
    })

@app.route('/api/reviews', methods=['GET'])
def api_get_reviews():
    category = request.args.get('category', None)
    search = request.args.get('search', None)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    reviews = get_reviews(category=category, search=search, limit=limit, offset=offset)
    return jsonify({'success': True, 'reviews': reviews})

@app.route('/api/reviews/<int:review_id>', methods=['DELETE'])
def api_delete_review(review_id):
    deleted = delete_review(review_id)
    return jsonify({'success': deleted})

@app.route('/api/reviews/clear', methods=['POST'])
def api_clear_reviews():
    clear_all_reviews()
    return jsonify({'success': True})

@app.route('/api/reviews/stats', methods=['GET'])
def api_get_stats():
    stats = get_review_stats()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/categories', methods=['GET'])
def api_get_categories():
    return jsonify({'success': True, 'categories': CATEGORIES})

def classify_into_six_buckets(r):
    cat = r.get('review_category', '')
    rating = int(r.get('rating', 5))
    text = (r.get('review_text') or '').lower()
    metrics = r.get('ml_metrics') or r.get('astar_metrics') or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except:
            metrics = {}

    is_ai = metrics.get('is_ai_generated', False)
    if not is_ai:
        ai_phrases = ["in conclusion", "testament to", "game-changer", "stands out as", "whether you are", "delving into", "boasts an impressive", "overall, i highly recommend", "a must-have"]
        if sum(1 for p in ai_phrases if p in text) >= 2:
            is_ai = True

    if "Genuine" in cat:
        return "genuine_good" if rating >= 4 else "genuine_bad"
    elif "Promotional" in cat:
        return "promo_spam"
    elif "Defamatory" in cat:
        return "defamatory"
    else: # Bot
        return "ai_generated" if is_ai else "bot_repetitive"

def get_all_consolidated_reviews():
    db_reviews = get_reviews(limit=500)
    brand_reviews = get_processed_brand_reviews()
    seen = set()
    combined = []
    for r in list(db_reviews) + list(brand_reviews):
        key = (r.get('product_name', '').strip().lower(), r.get('review_text', '').strip()[:80].lower())
        if key not in seen:
            seen.add(key)
            combined.append(r)
    return combined

@app.route('/api/reviews/distribution', methods=['GET'])
def api_get_distribution():
    product_name = request.args.get('product_name', '').strip()
    brand = request.args.get('brand', '').strip()

    reviews = get_all_consolidated_reviews()

    if product_name and product_name.lower() not in ['all', 'current']:
        reviews = [r for r in reviews if product_name.lower() in r.get('product_name', '').lower()]
    elif brand and brand.lower() != 'all':
        reviews = [r for r in reviews if brand.lower() in r.get('brand', '').lower()]

    counts = {
        "genuine_good": 0,
        "genuine_bad": 0,
        "promo_spam": 0,
        "defamatory": 0,
        "bot_repetitive": 0,
        "ai_generated": 0
    }

    for r in reviews:
        bucket = classify_into_six_buckets(r)
        counts[bucket] += 1

    total = sum(counts.values()) or 1
    percentages = {k: round((v / total) * 100.0, 1) for k, v in counts.items()}

    labels = {
        "genuine_good": "✅ 1. Genuine (Truthful) ➔ Good Product",
        "genuine_bad": "❌ 2. Genuine (Truthful) ➔ Bad Product",
        "promo_spam": "📢 3. Promotional Spam (Paid / Overhyped)",
        "defamatory": "⚔️ 4. Defamatory Spam (Competitor Smear)",
        "bot_repetitive": "🤖 5. Bot / Non-Review Junk (Repetitive)",
        "ai_generated": "🤖 6. AI-Generated Review (ChatGPT / Claude)"
    }

    return jsonify({
        'success': True,
        'product_name': product_name or 'All Products',
        'total_reviews': len(reviews),
        'counts': counts,
        'percentages': percentages,
        'labels': labels
    })

@app.route('/api/products/list', methods=['GET'])
def api_get_products_list():
    reviews = get_all_consolidated_reviews()
    unique_products = sorted(list(set(r.get('product_name', '').strip() for r in reviews if r.get('product_name'))))
    return jsonify({'success': True, 'products': unique_products})


# ==============================================================
# FAMOUS BRANDS REVIEWS & SORTING ALGORITHMS (LOCAL SEARCH / GA)
# ==============================================================

@app.route('/api/brands/reviews', methods=['GET'])
def api_get_brand_reviews():
    brand = request.args.get('brand', 'all')
    all_reviews = get_processed_brand_reviews()

    if brand and brand.lower() != 'all':
        filtered = [r for r in all_reviews if r['brand'].lower() == brand.lower()]
    else:
        filtered = list(all_reviews)

    return jsonify({
        'success': True,
        'brand': brand,
        'count': len(filtered),
        'reviews': filtered
    })

@app.route('/api/brands/sort', methods=['POST'])
def api_sort_brand_reviews():
    data = request.get_json() or {}
    brand = data.get('brand', 'all')
    algorithm = data.get('algorithm', 'genetic') # 'genetic' or 'local_search'

    all_reviews = get_processed_brand_reviews()
    if brand and brand.lower() != 'all':
        targets = [r for r in all_reviews if r['brand'].lower() == brand.lower()]
    else:
        targets = list(all_reviews)

    if not targets:
        return jsonify({'success': False, 'error': 'No reviews found to sort.'}), 400

    if algorithm == 'genetic':
        pop_size = int(data.get('pop_size', 40))
        generations = int(data.get('generations', 60))
        mutation_rate = float(data.get('mutation_rate', 0.20))
        result = run_genetic_algorithm_sorting(
            reviews=targets,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
    else: # local_search
        max_iterations = int(data.get('max_iterations', 80))
        result = run_local_search_sorting(
            reviews=targets,
            max_iterations=max_iterations
        )

    return jsonify(result)

@app.route('/api/brands/list', methods=['GET'])
def api_get_brands_list():
    all_reviews = get_processed_brand_reviews()
    known = ['Apple', 'Sony', 'Nike', 'Samsung', 'Dyson']
    present = sorted(list(set(r['brand'] for r in all_reviews if r.get('brand'))))
    ordered = [b for b in known if b in present] + [b for b in present if b not in known]
    return jsonify({'success': True, 'brands': ordered or known})

# ==============================================================
# CONSTRAINT SATISFACTION PROBLEM (CSP) REST APIS
# ==============================================================

@app.route('/api/csp/presets', methods=['GET'])
def api_csp_presets():
    return jsonify({'success': True, 'presets': PRESET_CSP_SCENARIOS})

@app.route('/api/csp/solve', methods=['POST'])
def api_csp_solve():
    data = request.get_json() or {}
    product_name = data.get('product_name', '').strip()
    brand = data.get('brand', 'all').strip()
    strict_mode = bool(data.get('strict_mode', True))

    reviews = get_all_consolidated_reviews()
    
    result = solve_top4_featured_csp(
        reviews=reviews,
        product_name=product_name if product_name and product_name.lower() != 'all' else None,
        brand=brand if brand and brand.lower() != 'all' else None,
        strict_mode=strict_mode
    )

    return jsonify(result)

@app.route('/api/csp/check_review', methods=['POST'])
def api_csp_check_review():
    data = request.get_json() or {}
    rating = int(data.get('rating', 5))
    review_text = data.get('review_text', '').strip()
    is_verified = bool(data.get('is_verified', False))
    
    is_congruent, norm_r, sent_val, div = check_sentiment_star_congruence(rating, review_text)
    aspects = extract_aspects(review_text)

    return jsonify({
        'success': True,
        'rating': rating,
        'norm_rating': norm_r,
        'sentiment_score': round(sent_val * 100, 1),
        'divergence': div,
        'is_congruent': is_congruent,
        'aspects': aspects,
        'is_verified': is_verified
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"=== Fake Product Review Detector with Supervised Machine Learning running on http://127.0.0.1:{port} ===")
    app.run(host='0.0.0.0', port=port, debug=False)
