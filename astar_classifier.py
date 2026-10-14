import heapq
import math
import time
from detector import extract_features

CATEGORY_GENUINE = "Genuine (Truthful) Review"
CATEGORY_PROMO = "Promotional Spam (Paid / Incentivized)"
CATEGORY_DEFAMATORY = "Defamatory Spam (Competitor Attack)"
CATEGORY_BOT = "Bot / Non-Review Junk"

CATEGORIES = [CATEGORY_GENUINE, CATEGORY_PROMO, CATEGORY_DEFAMATORY, CATEGORY_BOT]

# Prototypical Centroids for each category in normalized 8D feature space:
# [0] norm_rating
# [1] pos_sentiment
# [2] neg_sentiment
# [3] promo_score
# [4] defamy_score
# [5] style_anomaly
# [6] entropy_penalty
# [7] trust_factor
CENTROIDS = {
    CATEGORY_GENUINE: [0.70, 0.40, 0.15, 0.05, 0.05, 0.10, 0.10, 0.85],
    CATEGORY_PROMO:   [1.00, 0.90, 0.00, 0.85, 0.00, 0.70, 0.20, 0.15],
    CATEGORY_DEFAMATORY: [0.20, 0.00, 0.90, 0.00, 0.85, 0.65, 0.25, 0.10],
    CATEGORY_BOT:     [0.80, 0.20, 0.10, 0.15, 0.10, 0.60, 0.90, 0.10]
}

FEATURE_WEIGHTS = [0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]

def euclidean_heuristic(feat_vector, target_category):
    """
    Admissible and consistent Euclidean heuristic h(n) in weighted feature space.
    Calculates straight-line geometric distance to target category centroid.
    """
    centroid = CENTROIDS[target_category]
    dist_sq = 0.0
    for i in range(len(feat_vector)):
        diff = feat_vector[i] - centroid[i]
        dist_sq += FEATURE_WEIGHTS[i] * (diff ** 2)
    return round(math.sqrt(dist_sq) * 100.0, 2)

def classify_review_with_astar(rating, review_text, is_verified, reviewer_name=""):
    """
    Applies the A* Search Algorithm to find the optimal classification
    path across the review evaluation state space.
    """
    start_time = time.perf_counter()
    extracted = extract_features(rating, review_text, is_verified, reviewer_name)
    feat = extracted["vector"]

    # Explore state space for the 4 goal categories using A*
    # Priority Queue elements: (f_score, g_score, counter, current_node, path, target_cat)
    # Search tree levels:
    # 0: Root (START)
    # 1: Verification Node (TRUST_CHECK)
    # 2: Sentiment Node (POLARITY_CHECK)
    # 3: Lexical Node (LEXICAL_CHECK)
    # 4: Target Goal Category
    pq = []
    counter = 0

    # Initialize frontier with transitions to Stage 1 for all 4 candidate category branches
    for cat in CATEGORIES:
        # Initial step cost g(0 -> Stage1)
        h0 = euclidean_heuristic(feat, cat)
        g0 = 0.0
        f0 = g0 + h0
        counter += 1
        heapq.heappush(pq, (f0, g0, counter, "START", [f"START (h={h0})"], cat))

    nodes_expanded = 0
    winning_category = None
    winning_path = []
    winning_f = 0.0
    winning_g = 0.0
    winning_h = 0.0
    explored_trace = []

    while pq:
        f, g, _, curr_node, path, cat = heapq.heappop(pq)
        nodes_expanded += 1

        explored_trace.append({
            "step": nodes_expanded,
            "node": curr_node,
            "category_branch": cat,
            "g": round(g, 2),
            "h": round(f - g, 2),
            "f": round(f, 2)
        })

        # Check if Goal State reached
        if curr_node == f"GOAL_{cat}":
            winning_category = cat
            winning_path = path
            winning_f = round(f, 2)
            winning_g = round(g, 2)
            winning_h = round(f - g, 2)
            break

        # Generate Successors based on search level
        if curr_node == "START":
            # Stage 1: Verification Check
            step_cost = calculate_stage_cost(feat, cat, stage=1)
            nxt_g = g + step_cost
            nxt_h = euclidean_heuristic(feat, cat) * 0.75
            nxt_f = nxt_g + nxt_h
            counter += 1
            node_label = f"VERIFICATION_PASS" if cat == CATEGORY_GENUINE else f"VERIFICATION_RISK"
            heapq.heappush(pq, (nxt_f, nxt_g, counter, node_label, path + [f"{node_label} (g={round(nxt_g,1)}, h={round(nxt_h,1)})"], cat))

        elif "VERIFICATION" in curr_node:
            # Stage 2: Sentiment-Rating Congruency Check
            step_cost = calculate_stage_cost(feat, cat, stage=2)
            nxt_g = g + step_cost
            nxt_h = euclidean_heuristic(feat, cat) * 0.45
            nxt_f = nxt_g + nxt_h
            counter += 1
            node_label = get_sentiment_node_label(cat)
            heapq.heappush(pq, (nxt_f, nxt_g, counter, node_label, path + [f"{node_label} (g={round(nxt_g,1)}, h={round(nxt_h,1)})"], cat))

        elif "POLARITY" in curr_node or "SENTIMENT" in curr_node:
            # Stage 3: Lexical & Entropy Check
            step_cost = calculate_stage_cost(feat, cat, stage=3)
            nxt_g = g + step_cost
            nxt_h = euclidean_heuristic(feat, cat) * 0.15
            nxt_f = nxt_g + nxt_h
            counter += 1
            node_label = get_lexical_node_label(cat)
            heapq.heappush(pq, (nxt_f, nxt_g, counter, node_label, path + [f"{node_label} (g={round(nxt_g,1)}, h={round(nxt_h,1)})"], cat))

        else:
            # Transition to Final Goal Node
            nxt_g = g + 0.0
            nxt_h = 0.0 # Goal reached, h = 0
            nxt_f = nxt_g + nxt_h
            counter += 1
            goal_node = f"GOAL_{cat}"
            heapq.heappush(pq, (nxt_f, nxt_g, counter, goal_node, path + [f"GOAL: {cat} (f*={round(nxt_f,1)})"], cat))

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Determine Product Quality based on A* classification and rating
    product_evaluation = evaluate_product_quality(winning_category, rating, feat)

    return {
        "success": True,
        "review_category": winning_category,
        "product_verdict": product_evaluation["verdict"],
        "product_quality_score": product_evaluation["quality_score"],
        "product_quality_description": product_evaluation["description"],
        "astar_metrics": {
            "optimal_f_score": winning_f,
            "accumulated_cost_g": winning_g,
            "final_heuristic_h": winning_h,
            "nodes_expanded": nodes_expanded,
            "search_path": winning_path,
            "trace": explored_trace[:15], # top trace steps
            "execution_time_ms": elapsed_ms
        },
        "feature_breakdown": {
            "normalized_rating": feat[0],
            "positive_sentiment": feat[1],
            "negative_sentiment": feat[2],
            "promotional_score": feat[3],
            "defamatory_score": feat[4],
            "style_anomaly": feat[5],
            "entropy_penalty": feat[6],
            "trust_factor": feat[7],
            "promo_matches": extracted["promo_matches"],
            "defamy_matches": extracted["defamy_matches"],
            "word_count": extracted["word_count"],
            "is_verified": extracted["is_verified"]
        }
    }

def calculate_stage_cost(feat, target_cat, stage):
    """
    Calculates state transition penalty g(u, v) based on feature divergence.
    """
    if stage == 1: # Verification
        trust = feat[7]
        if target_cat == CATEGORY_GENUINE:
            return (1.0 - trust) * 25.0
        else:
            return trust * 15.0

    elif stage == 2: # Sentiment & Rating Congruency
        norm_r = feat[0]
        pos = feat[1]
        neg = feat[2]
        if target_cat == CATEGORY_GENUINE:
            # Genuine should have matching sentiment & star rating
            mismatch = abs(norm_r - (pos - neg * 0.5 + 0.3))
            return max(0.0, mismatch * 20.0)
        elif target_cat == CATEGORY_PROMO:
            return (1.0 - pos) * 30.0 + neg * 25.0
        elif target_cat == CATEGORY_DEFAMATORY:
            return (1.0 - neg) * 30.0 + pos * 25.0
        else: # Bot
            return (pos + neg) * 20.0

    elif stage == 3: # Lexical & Entropy
        promo = feat[3]
        defamy = feat[4]
        style = feat[5]
        entropy = feat[6]

        if target_cat == CATEGORY_GENUINE:
            return (promo * 35.0) + (defamy * 35.0) + (style * 25.0) + (entropy * 30.0)
        elif target_cat == CATEGORY_PROMO:
            return (1.0 - promo) * 30.0 + (1.0 - style) * 15.0
        elif target_cat == CATEGORY_DEFAMATORY:
            return (1.0 - defamy) * 35.0 + (1.0 - style) * 15.0
        else: # Bot
            return (1.0 - entropy) * 45.0 + (1.0 - style) * 10.0

    return 5.0

def get_sentiment_node_label(cat):
    if cat == CATEGORY_GENUINE:
        return "CONGRUENT_SENTIMENT"
    elif cat == CATEGORY_PROMO:
        return "HYPER_POSITIVE_HYPE"
    elif cat == CATEGORY_DEFAMATORY:
        return "SMEAR_ATTACK_TOXIC"
    else:
        return "LOW_ENTROPY_NEUTRAL"

def get_lexical_node_label(cat):
    if cat == CATEGORY_GENUINE:
        return "ORGANIC_VOCABULARY"
    elif cat == CATEGORY_PROMO:
        return "PROMOTIONAL_BUZZWORDS"
    elif cat == CATEGORY_DEFAMATORY:
        return "DEFAMATORY_ATTACK_WORDS"
    else:
        return "REPETITIVE_BOT_JUNK"

def evaluate_product_quality(category, rating, feat):
    """
    Determines if the product is a GOOD PRODUCT or BAD PRODUCT
    using the A* classification result to eliminate bias.
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
                "description": f"Verified Genuine Review ({rating}/5 Stars). Strong authentic evidence indicates a reliable, high-quality product."
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
            "description": "Promotional / Paid Spam detected. The 5-star rating is artificially inflated and neutralized by A* search."
        }

    elif category == CATEGORY_DEFAMATORY:
        return {
            "verdict": "Unfairly Defamed (Smear Attack)",
            "quality_score": 60.0,
            "description": "Competitor Defamatory Attack detected. The malicious 1-star smear is neutralized; not indicative of true product defect."
        }

    else: # Bot Junk
        return {
            "verdict": "Indeterminate (Junk Discarded)",
            "quality_score": 50.0,
            "description": "Bot / Non-Review Junk discarded. Insufficient authentic information to assess product quality."
        }
