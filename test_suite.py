import urllib.request
import json

BASE = 'http://127.0.0.1:5000'

def test_api(name, path, method='GET', payload=None):
    url = f'{BASE}{path}'
    data = json.dumps(payload).encode() if payload else None
    headers = {'Content-Type': 'application/json'} if payload else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"[OK] {name}: status={resp.status}, success={res.get('success')}")
        return res

if __name__ == '__main__':
    print("=== TESTING SUPERVISED ML REVIEW DETECTOR & BRAND REVIEW SORTING (GA/LOCAL SEARCH) ===\n")

    # 1. Test Supervised ML (Genuine Good Product)
    res1 = test_api(
        "1. Supervised ML: Genuine Good Product",
        "/api/reviews/analyze",
        "POST",
        {
            "product_name": "StudioPro Headphones",
            "rating": 5,
            "review_text": "Balanced frequency response and clear sound separation across all frequencies. Comfortable headband for daily audio engineering.",
            "is_verified": True,
            "reviewer_name": "Mark D."
        }
    )
    assert res1['review_category'] == "Genuine (Truthful) Review"
    print(f"   Category: {res1['review_category']} | Verdict: {res1['product_verdict']} | Conf: {res1['ml_metrics']['confidence_score']}%\n")

    # 2. Test Supervised ML (Promotional Spam)
    res2 = test_api(
        "2. Supervised ML: Promotional Spam",
        "/api/reviews/analyze",
        "POST",
        {
            "product_name": "Miracle Serum",
            "rating": 5,
            "review_text": "AMAZING PRODUCT BEST EVER IN THE WORLD! BUY NOW 100% GUARANTEED SATISFACTION MIRACLE EFFECT! USE DISCOUNT CODE BUY TODAY FAST!!",
            "is_verified": False,
            "reviewer_name": "BotPromo"
        }
    )
    assert res2['review_category'] == "Promotional Spam (Paid / Incentivized)"
    print(f"   Category: {res2['review_category']} | Verdict: {res2['product_verdict']} | Conf: {res2['ml_metrics']['confidence_score']}%\n")

    # 3. Test Supervised ML (Defamatory Attack)
    res3 = test_api(
        "3. Supervised ML: Defamatory Attack",
        "/api/reviews/analyze",
        "POST",
        {
            "product_name": "Vortex Blender",
            "rating": 1,
            "review_text": "BUY FROM BLENDTEC INSTEAD! TOTAL SCAM COMPANY STEALING MONEY CRIMINAL FRAUD BOYCOTT THIS PRODUCT IMMEDIATELY!! WORST TRASH EVER!!",
            "is_verified": False,
            "reviewer_name": "CompetitorShill"
        }
    )
    assert res3['review_category'] == "Defamatory Spam (Competitor Attack)"
    print(f"   Category: {res3['review_category']} | Verdict: {res3['product_verdict']} | Conf: {res3['ml_metrics']['confidence_score']}%\n")

    # 4. Test Supervised ML (AI-Generated ChatGPT Review Detection)
    res4 = test_api(
        "4. Supervised ML: AI-Generated Review (ChatGPT / Claude)",
        "/api/reviews/analyze",
        "POST",
        {
            "product_name": "Apple iPhone 16 Pro Max",
            "rating": 5,
            "review_text": "In conclusion, the Apple iPhone 16 Pro Max is a true testament to modern engineering. It stands out as a game-changer in the smartphone landscape. From its sleek and stylish titanium chassis to the seamless integration of Apple Intelligence, this device boasts an impressive display and stellar battery life. Furthermore, whether you are a casual user or a tech enthusiast, this phone offers something for everyone. Overall, I highly recommend this incredible piece of technology. A must-have addition to any daily routine!",
            "is_verified": False,
            "reviewer_name": "chatgpt_bot"
        }
    )
    assert res4['review_category'] == "Bot / Non-Review Junk"
    assert res4['ml_metrics']['is_ai_generated'] is True
    print(f"   Category: {res4['review_category']} | Verdict: {res4['product_verdict']}")
    print(f"   AI Score: {res4['ml_metrics']['ai_score']}% | Detected AI Phrases: {len(res4['ml_metrics']['ai_matches'])} matches")
    print(f"   Matched Cliches: {res4['ml_metrics']['ai_matches'][:4]}...\n")

    # 5. Test Auto-Adding a Brand Review from Tab 1 into Tab 2 (Nike test)
    res5 = test_api(
        "5. Auto-Routing: Add Nike Review in Tab 1 -> Verified in Brand Review Tab 2",
        "/api/reviews",
        "POST",
        {
            "product_name": "Nike Invincible 3 Running Shoes",
            "brand": "Nike",
            "category": "Footwear",
            "reviewer_name": "Sub4_Runner_Jake",
            "rating": 5,
            "review_text": "ZoomX cushion stack prevents leg fatigue on 20-mile Sunday training runs. Heel collar is secure.",
            "is_verified": True
        }
    )
    assert res5['added_to_brand_tab'] is True
    assert res5['brand'] == "Nike"
    print(f"   Added to Brand Tab: {res5['added_to_brand_tab']} | Brand: {res5['brand']} | Message: {res5['brand_message']}\n")

    # 6. Verify newly added Nike review appears in Brand Reviews list
    nike_reviews = test_api("6. Fetch Brand Reviews for Nike (Including newly auto-routed review)", "/api/brands/reviews?brand=Nike")
    found_new = any(r['product_name'] == "Nike Invincible 3 Running Shoes" for r in nike_reviews['reviews'])
    assert found_new is True
    print(f"   Nike Reviews Count: {nike_reviews['count']} | Found newly routed review: {found_new}\n")

    # 7. Test Database Scan History Stats
    stats = test_api("7. SQLite Review Scan History Stats", "/api/reviews/stats")
    print(f"   Corpus Stats: {stats['stats']}\n")

    # 8. Test Genetic Algorithm Sorting on Brand Reviews
    ga_res = test_api(
        "8. Genetic Algorithm (GA) Brand Review Ranking Sort",
        "/api/brands/sort",
        "POST",
        {
            "brand": "all",
            "algorithm": "genetic",
            "pop_size": 30,
            "generations": 40,
            "mutation_rate": 0.20
        }
    )
    print(f"   Initial Fitness: {ga_res['initial_fitness']} -> Final Fitness: {ga_res['final_fitness']} (Gain: +{ga_res['fitness_improvement']}) | Time: {ga_res['execution_time_ms']}ms")
    print(f"   Top Ranked Review: #{ga_res['sorted_reviews'][0]['id']} {ga_res['sorted_reviews'][0]['product_name']} by {ga_res['sorted_reviews'][0]['reviewer_name']} [Trust: {ga_res['sorted_reviews'][0]['trust_score']}]\n")

    # 9. Test Local Search Sorting on Brand Reviews
    local_res = test_api(
        "9. Local Search (Hill Climbing) Brand Review Ranking Sort",
        "/api/brands/sort",
        "POST",
        {
            "brand": "Apple",
            "algorithm": "local_search",
            "max_iterations": 50
        }
    )
    print(f"   Initial Fitness: {local_res['initial_fitness']} -> Final Fitness: {local_res['final_fitness']} (Gain: +{local_res['fitness_improvement']}) | Time: {local_res['execution_time_ms']}ms")
    print(f"   Top Ranked Apple Review: #{local_res['sorted_reviews'][0]['id']} by {local_res['sorted_reviews'][0]['reviewer_name']} [Trust: {local_res['sorted_reviews'][0]['trust_score']}]\n")

    # 10. Test 6-Category Review Distribution for Product Name
    dist = test_api("10. 6-Category Review Distribution for Product Name", "/api/reviews/distribution?product_name=Apple%20iPhone%2016%20Pro%20Max")
    print(f"   Product: {dist['product_name']} | Total Reviews: {dist['total_reviews']}")
    print(f"   6-Class Breakdown: {dist['percentages']}\n")

    # 11. Test Products List
    prod_list = test_api("11. Products List for Graph Selection", "/api/products/list")
    print(f"   Available Products Count: {len(prod_list['products'])}\n")

    # 12. Test CSP Presets API
    csp_presets = test_api("12. CSP Presets Showcase List", "/api/csp/presets")
    assert 'apple_iphone' in csp_presets['presets']
    print(f"   Loaded Presets: {list(csp_presets['presets'].keys())}\n")

    # 13. Test CSP Solver for Apple iPhone (All 6 Constraints Check)
    csp_res = test_api(
        "13. CSP Solver: Apple iPhone 16 Pro (Top-4 Featured Showcase)",
        "/api/csp/solve",
        "POST",
        {
            "brand": "Apple",
            "product_name": "Apple iPhone 16 Pro Max",
            "strict_mode": True
        }
    )
    assert csp_res['success'] is True
    assert len(csp_res['featured_slots']) == 4
    assert csp_res['constraints_status']['verified_buyer_protection']['satisfied'] is True
    assert csp_res['constraints_status']['copy_paste_bot_farm']['satisfied'] is True
    assert csp_res['constraints_status']['synchronized_burst_attack']['satisfied'] is True
    assert csp_res['constraints_status']['sentiment_star_contradiction']['satisfied'] is True
    assert csp_res['constraints_status']['top_4_featured_reviews']['satisfied'] is True
    assert csp_res['constraints_status']['complete_aspect_coverage']['satisfied'] is True
    print(f"   [CSP Check] All 6 Constraints Satisfied: {csp_res['all_constraints_satisfied']}")
    print(f"   [CSP Slots] 4 Slots Assigned: {[f['reviewer_name'] for f in csp_res['featured_slots']]}")
    print(f"   [CSP Aspects] Covered ({csp_res['aspect_coverage']['covered_count']}/4): {csp_res['aspect_coverage']['covered_aspects']}")
    print(f"   [CSP Trace] Explored: {csp_res['csp_trace']['nodes_explored']} | FC Prunings: {csp_res['csp_trace']['pruned_forward']} | Time: {csp_res['csp_trace']['execution_time_ms']}ms\n")

    # 14. Test CSP Solver for Nike & Sony
    nike_csp = test_api("14. CSP Solver: Nike Running Shoes Showcase", "/api/csp/solve", "POST", {"brand": "Nike"})
    assert len(nike_csp['featured_slots']) == 4
    print(f"   Nike Top-4 Slots: {[f['reviewer_name'] for f in nike_csp['featured_slots']]} | Aspects: {nike_csp['aspect_coverage']['covered_aspects']}\n")

    # 15. Test CSP Review Checker (Sentiment–Star Congruence & Aspects)
    chk1 = test_api(
        "15. CSP Review Checker: Congruent Review",
        "/api/csp/check_review",
        "POST",
        {
            "rating": 5,
            "review_text": "Exceptional battery life and fast speed with titanium build quality.",
            "is_verified": True
        }
    )
    assert chk1['is_congruent'] is True
    assert len(chk1['aspects']) >= 2
    print(f"   Congruent Test: is_congruent={chk1['is_congruent']} | Divergence={chk1['divergence']} | Aspects={chk1['aspects']}")

    chk2 = test_api(
        "16. CSP Review Checker: Contradiction Detection (5 Stars on Negative Text)",
        "/api/csp/check_review",
        "POST",
        {
            "rating": 5,
            "review_text": "Terrible broken garbage scam company completely defective waste.",
            "is_verified": False
        }
    )
    assert chk2['is_congruent'] is False
    print(f"   Contradiction Test: is_congruent={chk2['is_congruent']} | Divergence={chk2['divergence']} (Rejected by CSP)\n")

    # 17. Test Batch URL Review Scraper & Forensics
    scraped = test_api(
        "17. Batch URL Review Scraper: Extract & Forensics Audit",
        "/api/reviews/scrape_url",
        "POST",
        {
            "url": "https://www.apple.com/shop/buy-iphone/iphone-16-pro",
            "auto_save": True
        }
    )
    assert scraped['success'] is True
    assert scraped['total_scraped'] >= 5
    assert 'category_counts' in scraped
    assert 'overall_quality_score' in scraped
    print(f"   Batch Scraped: {scraped['total_scraped']} comments | Product: {scraped['product_name']}")
    print(f"   Overall Quality Score: {scraped['overall_quality_score']}% | Verdict: {scraped['overall_verdict']}")
    print(f"   Breakdown: {scraped['category_counts']}\n")

    print("===> ALL SUPERVISED ML, 6-CATEGORY GRAPH, CSP SOLVER, AND BATCH URL SCRAPER TESTS COMPLETED AND PASSED WITH 100% SUCCESS! <===")

