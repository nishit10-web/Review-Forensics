import math
import random
import time

def calculate_review_trust_score(review):
    """
    Computes an objective trust and informational utility score for a review
    based on its A* categorization and feature signals:
    - Genuine (Truthful) Review: 85 - 100 points
    - Average/Critical Genuine: 70 - 85 points
    - Artificially Overhyped (Promotional): 20 - 35 points
    - Defamatory Smear Attack: 15 - 30 points
    - Bot / Non-Review Junk: 0 - 15 points
    """
    cat = review.get("review_category", "")
    rating = int(review.get("rating", 3))
    is_verified = bool(review.get("is_verified", False))
    word_count = len(review.get("review_text", "").split())

    if cat == "Genuine (Truthful) Review":
        base = 80.0
        if is_verified: base += 10.0
        if 20 <= word_count <= 150: base += 10.0 # optimal informative length
        return min(100.0, base)
    elif cat == "Promotional Spam (Paid / Incentivized)":
        return 25.0
    elif cat == "Defamatory Spam (Competitor Attack)":
        return 20.0
    else: # Bot / Junk
        return 8.0

def evaluate_ranking_fitness(tour, reviews):
    """
    Discounted Cumulative Gain (DCG) style fitness function for review ranking:
    Ranks that place high-trust, genuine reviews near the top receive maximum fitness.
    Ranks that place spam or bot junk in top slots are heavily penalized.
    """
    fitness = 0.0
    n = len(tour)

    for rank, idx in enumerate(tour):
        r = reviews[idx]
        trust = calculate_review_trust_score(r)
        
        # Position discount weight (higher rank = higher weight, like search engine results)
        pos_weight = 1.0 / math.log2(rank + 2) # rank 0 has weight 1.0, rank 1 has weight 0.63, etc.
        
        # Add discounted trust score
        fitness += trust * pos_weight

        # Heavy penalty if top 3 reviews are spam or bot junk
        cat = r.get("review_category", "")
        if rank < 3 and cat != "Genuine (Truthful) Review":
            fitness -= (4 - rank) * 20.0

    return round(fitness, 2)

# ==============================================================
# 1. GENETIC ALGORITHM FOR REVIEW RANKING OPTIMIZATION
# ==============================================================

def ordered_crossover_ranking(p1, p2):
    n = len(p1)
    if n <= 2: return list(p1)
    idx1, idx2 = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[idx1:idx2+1] = p1[idx1:idx2+1]

    p2_idx = (idx2 + 1) % n
    ch_idx = (idx2 + 1) % n
    while -1 in child:
        cand = p2[p2_idx]
        if cand not in child:
            child[ch_idx] = cand
            ch_idx = (ch_idx + 1) % n
        p2_idx = (p2_idx + 1) % n
    return child

def mutate_swap_ranking(tour, mutation_rate):
    if random.random() < mutation_rate and len(tour) > 1:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def run_genetic_algorithm_sorting(reviews, pop_size=40, generations=60, mutation_rate=0.20):
    start_time = time.perf_counter()
    n = len(reviews)
    if n <= 1:
        return {
            "success": True,
            "sorted_reviews": reviews,
            "fitness_history": [{"generation": 1, "best_fitness": 100, "avg_fitness": 100}],
            "initial_fitness": 100,
            "final_fitness": 100,
            "execution_time_ms": 0.1
        }

    # Initial random population of review permutations
    base_indices = list(range(n))
    population = [random.sample(base_indices, n) for _ in range(pop_size)]

    # Include original order in initial population
    population[0] = list(base_indices)

    initial_fitness = evaluate_ranking_fitness(base_indices, reviews)
    fitness_history = []
    best_overall_tour = None
    best_overall_fitness = -math.inf

    checkpoints = []
    checkpoint_step = max(1, generations // 10)

    for gen in range(generations):
        fitnesses = [evaluate_ranking_fitness(ind, reviews) for ind in population]
        max_fit = max(fitnesses)
        best_idx = fitnesses.index(max_fit)
        avg_fit = sum(fitnesses) / len(fitnesses)

        if max_fit > best_overall_fitness:
            best_overall_fitness = max_fit
            best_overall_tour = list(population[best_idx])

        fitness_history.append({
            "generation": gen + 1,
            "best_fitness": round(max_fit, 2),
            "avg_fitness": round(avg_fit, 2)
        })

        if gen % checkpoint_step == 0 or gen == generations - 1:
            checkpoints.append({
                "generation": gen + 1,
                "best_fitness": round(max_fit, 2),
                "tour": list(population[best_idx])
            })

        # Selection & Elitism
        sorted_by_fit = sorted(range(pop_size), key=lambda k: fitnesses[k], reverse=True)
        new_pop = [list(population[sorted_by_fit[0]])]
        if pop_size > 1:
            new_pop.append(list(population[sorted_by_fit[1]]))

        def tournament():
            k_cand = random.sample(range(pop_size), 3)
            best_k = max(k_cand, key=lambda idx: fitnesses[idx])
            return population[best_k]

        while len(new_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child = ordered_crossover_ranking(p1, p2)
            child = mutate_swap_ranking(child, mutation_rate)
            new_pop.append(child)

        population = new_pop

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    sorted_reviews = [reviews[i] for i in best_overall_tour]

    return {
        "success": True,
        "algorithm": "Genetic Algorithm (GA)",
        "sorted_reviews": sorted_reviews,
        "initial_fitness": initial_fitness,
        "final_fitness": best_overall_fitness,
        "fitness_improvement": round(best_overall_fitness - initial_fitness, 2),
        "fitness_history": fitness_history,
        "checkpoints": checkpoints,
        "generations": generations,
        "pop_size": pop_size,
        "execution_time_ms": elapsed_ms
    }

# ==============================================================
# 2. LOCAL SEARCH ALGORITHM (HILL CLIMBING)
# ==============================================================

def run_local_search_sorting(reviews, max_iterations=80):
    start_time = time.perf_counter()
    n = len(reviews)
    if n <= 1:
        return {
            "success": True,
            "sorted_reviews": reviews,
            "fitness_history": [{"iteration": 1, "fitness": 100}],
            "initial_fitness": 100,
            "final_fitness": 100,
            "execution_time_ms": 0.1
        }

    current_tour = list(range(n))
    current_fitness = evaluate_ranking_fitness(current_tour, reviews)
    initial_fitness = current_fitness

    history = [{
        "iteration": 0,
        "fitness": current_fitness
    }]

    swaps_made = 0

    for it in range(1, max_iterations + 1):
        # Generate 2-swap neighbors
        best_neighbor = None
        best_neighbor_fit = current_fitness

        # Sample neighborhood swaps
        candidates = []
        for _ in range(min(40, n * (n - 1) // 2)):
            i, j = random.sample(range(n), 2)
            neighbor = list(current_tour)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            fit = evaluate_ranking_fitness(neighbor, reviews)
            if fit > best_neighbor_fit:
                best_neighbor_fit = fit
                best_neighbor = neighbor

        if best_neighbor and best_neighbor_fit > current_fitness:
            current_tour = best_neighbor
            current_fitness = best_neighbor_fit
            swaps_made += 1

        history.append({
            "iteration": it,
            "fitness": current_fitness
        })

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    sorted_reviews = [reviews[i] for i in current_tour]

    return {
        "success": True,
        "algorithm": "Local Search (Hill Climbing)",
        "sorted_reviews": sorted_reviews,
        "initial_fitness": initial_fitness,
        "final_fitness": current_fitness,
        "fitness_improvement": round(current_fitness - initial_fitness, 2),
        "fitness_history": history,
        "iterations": max_iterations,
        "swaps_made": swaps_made,
        "execution_time_ms": elapsed_ms
    }
