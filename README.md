# Fake Product Review Detector & Supervised Machine Learning Quality Evaluator
## Featuring Brand Review Ranking (GA / Local Search) & Constraint Satisfaction Problem (CSP)

A comprehensive AI-driven web application featuring:
1. **Fake Product Review Detection & Supervised ML Product Quality Evaluation**:
   - Multi-class categorization:
     - **✅ Genuine (Truthful) Review** (Good Product vs Bad Product)
     - **📢 Promotional Spam (Paid / Overhyped)**
     - **⚔️ Defamatory Spam (Competitor Smear Attack)**
     - **🤖 Bot / Non-Review Junk (Repetitive Strings & AI/LLM Generated Text)**
   - **Interactive 6-Category Review Distribution Graph** for any selected Product Name.
   - **Review Scan History** with authenticity labels and algorithm performance scores (backed by SQLite `reviews.db`).
   - Dynamic auto-routing: reviews added for Apple, Nike, Sony, Samsung, Dyson in Tab 1 automatically sync to Tab 2 and Tab 3.

2. **Brand review (Rank & Sort via Local Search / Genetic Algorithm)**:
   - Dedicated interactive tab featuring iconic brands (**Apple, Sony, Nike, Samsung, Dyson**).
   - Re-ranks reviews to maximize **Discounted Cumulative Gain (DCG) Trust Fitness**:
     - **Genetic Algorithm (GA)**: Permutation evolution with Tournament Selection, Ordered Crossover (OX1), and Mutation.
     - **Local Search (Hill Climbing)**: Iterative 2-swap neighborhood exploration.
   - Real-time fitness convergence curves and category distribution charts.

3. **Constraint Satisfaction Problem (CSP) Showcase Selector**:
   - Solves the **Top-4 Featured Reviews** showcase selection problem as a formal CSP (\(X = \{S_1, S_2, S_3, S_4\}\), domain \(D(S_i)\)).
   - Implements **Backtracking Search with Forward Checking (FC)** and **Minimum Remaining Values (MRV)** heuristic.
   - Rigorously enforces all 6 explicit constraints:
     1. **Verified Buyer Protection**: Slots \(S_1\) and \(S_2\) MUST be verified buyers; \(\ge 3/4\) slots verified; unverified spam pruned.
     2. **Copy-Paste Bot Farm**: Pairwise Jaccard word similarity between assigned reviews must be \(< 0.35\); repetitive token loops eliminated.
     3. **Synchronized Burst Attack**: No two reviews in the same \(\le 30\)-minute burst window with matching smear/hype polarity; requires distinct reviewer handles.
     4. **Sentiment–Star Contradiction**: Divergence \(|(\text{Rating}/5.0) - \text{Sentiment}| \le 0.40\); severe rating-text mismatches pruned.
     5. **Top-4 Featured Reviews**: Exactly 4 unique slots populated under All-Different constraint (\(S_1 \ne S_2 \ne S_3 \ne S_4\)).
     6. **Complete Product Aspect Coverage**: Assigned reviews collectively cover \(\ge 3\) core aspects: *Performance & Speed*, *Build Quality & Durability*, *Comfort & Design*, *Battery & Thermals*.

4. **Batch URL Review Scraper & Forensics**:
   - Live URL extraction of customer comments using `BeautifulSoup4` with browser user-agent simulation.
   - Built-in adaptive fail-safe for anti-bot blocked domains (Apple, Nike, Sony, Samsung, Dyson).
   - Instant bulk ML classification, product quality scoring, and automated SQLite persistence.

---

## 📁 Project Architecture

```
fake_review_and_ai_suite/
├── app.py                  # Flask server and REST API endpoints (ML, GA, CSP, Scraper)
├── ml_classifier.py        # Supervised Machine Learning multi-class classifier
├── detector.py             # Feature extractor & ChatGPT/Claude AI synthetic review detector
├── url_scraper.py          # Batch URL scraper with BeautifulSoup & bulk ML audit pipeline
├── csp_solver.py           # Constraint Satisfaction Problem (CSP) solver (6 constraints, FC, MRV)
├── brands_dataset.py       # Official Brand reviews corpus (Apple, Sony, Nike, Samsung, Dyson)
├── database.py             # SQLite database schemas, queries, and persistence
├── sorting_algorithms.py   # Genetic Algorithm & Local Search (Hill Climbing) review sorters
├── test_suite.py           # Automated 17-test end-to-end verification script
├── templates/
│   └── index.html          # Web UI with 3 Tabs (Detector, Brand review, CSP Solver)
└── static/
    ├── style.css           # Clean, modern white-themed stylesheet
    └── main.js             # Frontend controller (ML, Canvas charts, GA, CSP showcase)
```

---

## 🚀 How to Run

### 1. Launch Server
```powershell
cd C:\Users\Nishit\.gemini\antigravity\scratch\fake_review_and_ai_suite
python app.py
```

### 2. Open in Browser
Visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/reviews/analyze` | `POST` | Supervised ML prediction, AI bot score, and 6-class breakdown |
| `/api/reviews` | `POST` | Evaluate review via ML, save to SQLite, and auto-route to Brand tab |
| `/api/reviews/distribution` | `GET` | 6-Category distribution counts & percentages for any product |
| `/api/brands/reviews` | `GET` | Fetch brand reviews (`?brand=Apple`, `Sony`, `Nike`, etc.) |
| `/api/brands/sort` | `POST` | Optimize brand reviews using Genetic Algorithm or Local Search |
| `/api/csp/solve` | `POST` | Solve CSP with Backtracking + Forward Checking across the 6 constraints |
| `/api/csp/presets` | `GET` | Return preset showcase scenarios (iPhone, XM5, Pegasus, S24, V15) |
| `/api/csp/check_review` | `POST` | Validate single review against Sentiment-Star congruence & extract aspects |
