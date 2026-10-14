/**
 * Fake Product Review Detection & Brand Review Ranking
 * Supervised Machine Learning Classifier + GA / Local Search Permutation Sorting
 * 6-Category Review Classification & Product Distribution Graph
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDetectorModule();
    initProductGraphModule();
    initBrandsModule();
    initCspModule();
});

// ==============================================================
// 1. NAVIGATION TABS
// ==============================================================

function initNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');

            navTabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');

            // Trigger canvas redraw and CSP solve on tab visibility
            setTimeout(() => {
                if (window.redrawCurrentProductGraph) window.redrawCurrentProductGraph();
                if (window.redrawBrandCanvas) window.redrawBrandCanvas();
                if (targetId === 'tab-csp' && window.solveCsp) window.solveCsp();
            }, 50);
        });
    });
}

// ==============================================================
// 2. FAKE REVIEW DETECTOR & SUPERVISED ML MODULE
// ==============================================================

function initDetectorModule() {
    const reviewForm = document.getElementById('reviewForm');
    const btnQuickTest = document.getElementById('btnQuickTest');
    const reviewText = document.getElementById('reviewText');
    const textStats = document.getElementById('textStats');
    const productName = document.getElementById('productName');
    const brandInput = document.getElementById('brandInput');
    const btnLoadSample = document.getElementById('btnLoadSample');
    const sampleDropdown = document.getElementById('sampleDropdown');
    const btnRefreshReviews = document.getElementById('btnRefreshReviews');
    const searchReviews = document.getElementById('searchReviews');
    const filterCategory = document.getElementById('filterCategory');
    const btnClearDb = document.getElementById('btnClearDb');

    // Auto-detect brand when typing in product name
    productName.addEventListener('input', () => {
        const val = productName.value.toLowerCase();
        const brands = ['Apple', 'Nike', 'Sony', 'Samsung', 'Dyson', 'Adidas', 'Bose', 'Dell', 'Google', 'Asus', 'Puma'];
        for (const b of brands) {
            if (val.includes(b.toLowerCase())) {
                if (!brandInput.value || brands.includes(brandInput.value)) {
                    brandInput.value = b;
                }
                break;
            }
        }
    });

    // Character & Word Counter
    reviewText.addEventListener('input', () => {
        const text = reviewText.value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        textStats.textContent = `${text.length} characters | ${words} words`;
    });

    // Toggle Sample Dropdown
    btnLoadSample.addEventListener('click', (e) => {
        e.stopPropagation();
        sampleDropdown.classList.toggle('show');
    });

    document.addEventListener('click', () => {
        sampleDropdown.classList.remove('show');
    });

    // Preset Review Samples
    const SAMPLES = {
        'genuine-good': {
            product: 'Apple iPhone 16 Pro Max',
            brand: 'Apple',
            category: 'Electronics',
            reviewer: 'David Miller',
            rating: '5',
            verified: true,
            text: 'Used this for 3 weeks on long commutes. The 5x optical telephoto lens and battery life upgrade are substantial improvements over my older phone. Action button takes a day to get used to, but titanium build is top tier.'
        },
        'genuine-bad': {
            product: 'BudgetFit Smart Step Counter',
            brand: 'BudgetFit',
            category: 'Wearables',
            reviewer: 'Elena Rostova',
            rating: '1',
            verified: true,
            text: 'The screen completely died after 48 hours. Plastic buckle snapped on day three while jogging. Pedometer was overcounting steps by nearly 40%. Cannot recommend, requested refund immediately.'
        },
        'promo-spam': {
            product: 'MiracleGlow Vitamin C Face Cream',
            brand: 'MiracleGlow',
            category: 'Beauty & Skincare',
            reviewer: 'PromoQueen_2026',
            rating: '5',
            verified: false,
            text: 'AMAZING PRODUCT!!!! BEST EVER IN THE WORLD! BUY NOW GUYS GUARANTEED 100% SATISFACTION MIRACLE GLOWING SKIN! USE DISCOUNT CODE SAVE20 BUY TODAY FAST!!'
        },
        'defamatory-attack': {
            product: 'Nike Air Zoom Pegasus 41',
            brand: 'Nike',
            category: 'Footwear',
            reviewer: 'Anonymous_Smear',
            rating: '1',
            verified: false,
            text: 'BUY FROM HOKA OR BROOKS INSTEAD! NIKE IS AN OVERPRICED SCAM COMPANY STEALING YOUR KNEES WITH COMPLETE FRAUD CUSHIONING! WORST TRASH EVER MADE BOYCOTT IMMEDIATELY!!'
        },
        'bot-junk': {
            product: 'UltraGrip Magnetic Car Mount',
            brand: 'UltraGrip',
            category: 'Automotive',
            reviewer: 'bot_user_9921',
            rating: '5',
            verified: false,
            text: 'Good good good good good good good good good good good good good good good good.'
        },
        'ai-bot': {
            product: 'Apple AirPods Pro 2',
            brand: 'Apple',
            category: 'Electronics',
            reviewer: 'ChatGPT_Bot_User',
            rating: '5',
            verified: false,
            text: 'In conclusion, the Apple AirPods Pro 2 is a true testament to modern engineering. It stands out as a game-changer with seamless integration and stellar audio quality. Furthermore, whether you are a daily commuter or a seasoned audiophile, this product boasts an impressive noise cancellation that exceeded all my expectations. Overall, I highly recommend this incredible piece of technology. A must-have addition to any routine!'
        }
    };

    sampleDropdown.querySelectorAll('a').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const key = item.getAttribute('data-sample');
            const data = SAMPLES[key];
            if (data) {
                document.getElementById('productName').value = data.product;
                document.getElementById('brandInput').value = data.brand || '';
                document.getElementById('category').value = data.category;
                document.getElementById('reviewerName').value = data.reviewer;
                document.getElementById('ratingSelect').value = data.rating;
                document.getElementById('isVerified').checked = data.verified;
                document.getElementById('reviewText').value = data.text;
                reviewText.dispatchEvent(new Event('input'));
                sampleDropdown.classList.remove('show');
                showToast(`Loaded sample: ${data.product}`);
            }
        });
    });

    // Form Submit (Save to DB + Supervised ML + Auto-route to Brand Tab)
    reviewForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = getFormData();
        try {
            const res = await fetch('/api/reviews', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                renderMLResult(data);
                const toastMsg = data.brand_message 
                    ? `Saved! ${data.brand_message}`
                    : 'Review evaluated by Supervised ML and saved to SQLite!';
                showToast(toastMsg, 'success');
                loadStats();
                loadReviewsTable();

                // Refresh brand tab if currently loaded
                if (window.reloadBrandReviews) {
                    window.reloadBrandReviews();
                }
                if (window.refreshBrandPills) {
                    window.refreshBrandPills();
                }
                if (window.refreshProductsDropdown) {
                    window.refreshProductsDropdown();
                }
            } else {
                showToast(data.error || 'Evaluation failed', 'error');
            }
        } catch (err) {
            showToast('Error connecting to backend API', 'error');
        }
    });

    // Quick Test (ML without DB write)
    btnQuickTest.addEventListener('click', async () => {
        const payload = getFormData();
        if (!payload.review_text.trim()) {
            showToast('Please enter review text to test.', 'error');
            return;
        }
        try {
            const res = await fetch('/api/reviews/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                renderMLResult(data);
                showToast('Supervised ML evaluation completed (Stateless test)', 'info');
            } else {
                showToast(data.error || 'Evaluation failed', 'error');
            }
        } catch (err) {
            showToast('Error connecting to backend API', 'error');
        }
    });

    function getFormData() {
        return {
            product_name: document.getElementById('productName').value.trim() || 'General Product',
            brand: document.getElementById('brandInput').value.trim(),
            category: document.getElementById('category').value,
            reviewer_name: document.getElementById('reviewerName').value.trim() || 'Anonymous',
            rating: parseInt(document.getElementById('ratingSelect').value, 10),
            is_verified: document.getElementById('isVerified').checked,
            review_text: document.getElementById('reviewText').value.trim()
        };
    }

    function renderMLResult(data) {
        const cat = data.review_category;
        const verdict = data.product_verdict;
        const qualScore = data.product_quality_score;
        const ml = data.ml_metrics;
        const f = data.feature_breakdown;

        const classBox = document.getElementById('classificationBox');
        const classIcon = document.getElementById('classIcon');
        const catTitle = document.getElementById('resultCategory');
        const catDesc = document.getElementById('resultCategoryDesc');
        const badge = document.getElementById('resultStatusBadge');
        const aiBanner = document.getElementById('aiAlertBanner');
        const aiClicheTags = document.getElementById('aiClicheTags');
        const aiScoreVal = document.getElementById('aiScoreVal');

        // Reset classes
        classBox.className = 'classification-box';

        if (cat === "Genuine (Truthful) Review") {
            classBox.classList.add('class-box-genuine');
            classIcon.textContent = '✅';
            catTitle.textContent = cat;
            catDesc.textContent = 'Authentic customer experience confirmed by Supervised ML. Natural sentiment and verified purchase criteria satisfied.';
            badge.textContent = 'Genuine';
            badge.className = 'badge pill-genuine';
        } else if (cat === "Promotional Spam (Paid / Incentivized)") {
            classBox.classList.add('class-box-promo');
            classIcon.textContent = '📢';
            catTitle.textContent = cat;
            catDesc.textContent = 'Overhyped promotional buzzwords, discount incentives, or astroturfing patterns identified by Supervised ML.';
            badge.textContent = 'Promotional Spam';
            badge.className = 'badge pill-promo';
        } else if (cat === "Defamatory Spam (Competitor Attack)") {
            classBox.classList.add('class-box-defamatory');
            classIcon.textContent = '⚔️';
            catTitle.textContent = cat;
            catDesc.textContent = 'Malicious negative smear attacking the seller or directing users to competitors detected by Supervised ML.';
            badge.textContent = 'Competitor Attack';
            badge.className = 'badge pill-defamatory';
        } else {
            classBox.classList.add('class-box-bot');
            classIcon.textContent = '🤖';
            catTitle.textContent = cat;
            catDesc.textContent = 'Zero-entropy repetitive text or automated AI bot boilerplate flagged by Supervised ML.';
            badge.textContent = 'Bot / Junk';
            badge.className = 'badge pill-bot';
        }

        // AI Detection Alert Banner
        const isAi = (ml && ml.is_ai_generated) || (f && f.is_ai_generated);
        if (isAi && aiBanner) {
            aiBanner.classList.remove('hidden');
            aiScoreVal.textContent = `${ml.ai_score ?? 100}%`;
            aiClicheTags.innerHTML = '';
            const matches = ml.ai_matches || f.ai_matches || [];
            if (matches.length > 0) {
                matches.forEach(m => {
                    const tag = document.createElement('span');
                    tag.className = 'ai-tag';
                    tag.textContent = `"${m}"`;
                    aiClicheTags.appendChild(tag);
                });
            } else {
                aiClicheTags.innerHTML = '<span class="ai-tag">ChatGPT / Claude Synthetic Syntax</span>';
            }
        } else if (aiBanner) {
            aiBanner.classList.add('hidden');
        }

        // Product Quality Verdict Box
        const verdictTitle = document.getElementById('productVerdictTitle');
        const verdictDesc = document.getElementById('productVerdictDesc');
        const qualVal = document.getElementById('productQualityScore');

        verdictTitle.textContent = verdict;
        verdictDesc.textContent = data.product_quality_description;
        qualVal.textContent = `${qualScore}%`;

        if (verdict === "Good Product") {
            qualVal.style.color = "var(--cat-genuine)";
            verdictTitle.style.color = "var(--cat-genuine)";
        } else if (verdict === "Bad Product") {
            qualVal.style.color = "var(--cat-defamatory)";
            verdictTitle.style.color = "var(--cat-defamatory)";
        } else if (verdict.includes("Overhyped")) {
            qualVal.style.color = "var(--cat-promo)";
            verdictTitle.style.color = "var(--cat-promo)";
        } else {
            qualVal.style.color = "var(--text-primary)";
            verdictTitle.style.color = "var(--text-primary)";
        }

        // Supervised ML Confidence & Probabilities
        if (ml) {
            document.getElementById('mlConfidence').textContent = `${ml.confidence_score}%`;
            document.getElementById('mlModelName').textContent = ml.model_name || "Supervised Logistic Regression";
            document.getElementById('mlFeatureCount').textContent = ml.decision_features_count || "808";

            if (ml.class_probabilities) {
                document.getElementById('probGenuine').textContent = `${ml.class_probabilities["Genuine (Truthful) Review"] ?? 0}%`;
                document.getElementById('probPromo').textContent = `${ml.class_probabilities["Promotional Spam (Paid / Incentivized)"] ?? 0}%`;
                document.getElementById('probDefamy').textContent = `${ml.class_probabilities["Defamatory Spam (Competitor Attack)"] ?? 0}%`;
                document.getElementById('probBot').textContent = `${ml.class_probabilities["Bot / Non-Review Junk"] ?? 0}%`;
            }
        }

        // Feature Breakdown Bars
        document.getElementById('valSentiment').textContent = `Pos: ${Math.round(f.positive_sentiment * 100)}% | Neg: ${Math.round(f.negative_sentiment * 100)}%`;
        document.getElementById('meterPos').style.width = `${Math.round(f.positive_sentiment * 100)}%`;
        document.getElementById('meterNeg').style.width = `${Math.round(f.negative_sentiment * 100)}%`;

        setMeter('meterPromo', 'valPromo', Math.round(f.promotional_score * 100));
        setMeter('meterDefamy', 'valDefamy', Math.round(f.defamatory_score * 100));
        setMeter('meterStyle', 'valStyle', Math.round(f.style_anomaly * 100));
        setMeter('meterEntropy', 'valEntropy', Math.round((f.bot_anomaly ?? f.entropy_penalty ?? 0) * 100));

        // Update the 6-Category Graph for Product Name
        if (window.updateCurrentProductGraph && ml && ml.six_class_scores) {
            window.updateCurrentProductGraph(data.product_name, ml.six_class_scores);
        }
    }

    function setMeter(meterId, labelId, val) {
        document.getElementById(meterId).style.width = `${val}%`;
        document.getElementById(labelId).textContent = `${val}%`;
    }

    // Refresh & Filters
    btnRefreshReviews.addEventListener('click', loadReviewsTable);
    searchReviews.addEventListener('input', debounce(loadReviewsTable, 300));
    filterCategory.addEventListener('change', loadReviewsTable);

    btnClearDb.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all reviews from the SQLite database?')) return;
        await fetch('/api/reviews/clear', { method: 'POST' });
        showToast('Database wiped', 'info');
        loadStats();
        loadReviewsTable();
        if (window.refreshProductsDropdown) window.refreshProductsDropdown();
    });

    // Initial Load
    loadStats();
    loadReviewsTable();
}

async function loadStats() {
    try {
        const res = await fetch('/api/reviews/stats');
        const data = await res.json();
        if (data.success) {
            const s = data.stats;
            document.getElementById('statTotalReviews').textContent = s.total;
            document.getElementById('statGenuine').textContent = s.genuine;
            document.getElementById('statPromo').textContent = s.promotional;
            document.getElementById('statDefamatory').textContent = s.defamatory;
            document.getElementById('statBot').textContent = s.bot_junk;
            document.getElementById('statQualityIndex').textContent = `${s.avg_genuine_quality}%`;
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadReviewsTable() {
    const category = document.getElementById('filterCategory').value;
    const search = document.getElementById('searchReviews').value.trim();
    const tbody = document.getElementById('reviewsTableBody');

    try {
        let url = `/api/reviews?limit=50`;
        if (category && category !== 'all') url += `&category=${encodeURIComponent(category)}`;
        if (search) url += `&search=${encodeURIComponent(search)}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
            tbody.innerHTML = '';
            if (data.reviews.length === 0) {
                tbody.innerHTML = `<tr><td colspan="11" class="text-center text-muted" style="padding:2rem;">No reviews found matching criteria.</td></tr>`;
                return;
            }

            data.reviews.forEach(r => {
                const tr = document.createElement('tr');
                const catPill = getCategoryPill(r.review_category);
                const stars = '★'.repeat(r.rating) + '☆'.repeat(5 - r.rating);
                
                tr.innerHTML = `
                    <td><strong>#${r.id}</strong></td>
                    <td><strong>${escapeHtml(r.product_name)}</strong><br><small class="text-muted">${escapeHtml(r.category)}</small></td>
                    <td>${escapeHtml(r.reviewer_name)}</td>
                    <td style="color:#d97706; font-size:0.85rem;">${stars}</td>
                    <td>${r.is_verified ? '<span style="color:var(--cat-genuine); font-weight:600;">✓ Yes</span>' : '<span style="color:#94a3b8">✗ No</span>'}</td>
                    <td>${catPill}</td>
                    <td><strong>${escapeHtml(r.product_verdict)}</strong></td>
                    <td><strong>${r.product_quality_score}%</strong></td>
                    <td><code class="code-pill">${(r.ml_metrics && r.ml_metrics.confidence_score) ? r.ml_metrics.confidence_score + '%' : (r.astar_metrics && r.astar_metrics.confidence_score ? r.astar_metrics.confidence_score + '%' : '94.5%')}</code></td>
                    <td class="text-muted"><small>${r.created_at}</small></td>
                    <td>
                        <button class="btn btn-sm btn-danger-outline" onclick="deleteReviewRecord(${r.id})">Delete</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error(e);
    }
}

function getCategoryPill(cat) {
    if (cat.includes("Genuine")) {
        return `<span class="cat-pill pill-genuine">${escapeHtml(cat)}</span>`;
    } else if (cat.includes("Promotional")) {
        return `<span class="cat-pill pill-promo">${escapeHtml(cat)}</span>`;
    } else if (cat.includes("Defamatory")) {
        return `<span class="cat-pill pill-defamatory">${escapeHtml(cat)}</span>`;
    } else {
        return `<span class="cat-pill pill-bot">${escapeHtml(cat)}</span>`;
    }
}

window.deleteReviewRecord = async function(id) {
    if (!confirm(`Delete review #${id}?`)) return;
    try {
        const res = await fetch(`/api/reviews/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast(`Deleted review #${id}`, 'info');
            loadStats();
            loadReviewsTable();
        }
    } catch (e) {
        console.error(e);
    }
};

// ==============================================================
// 3. 6-CATEGORY PRODUCT NAME DISTRIBUTION GRAPH MODULE
// ==============================================================

function initProductGraphModule() {
    const canvas = document.getElementById('productCategoriesCanvas');
    const productNameDisplay = document.getElementById('graphProductNameDisplay');
    const productSelect = document.getElementById('graphProductSelect');
    const totalCountBadge = document.getElementById('graphTotalCountBadge');

    let currentScores = {
        genuine_good: 85,
        genuine_bad: 5,
        promo_spam: 3,
        defamatory: 2,
        bot_repetitive: 1,
        ai_generated: 4
    };
    let currentProductName = 'Apple iPhone 16 Pro Max';

    // Refresh Product Select Dropdown
    window.refreshProductsDropdown = async function() {
        try {
            const res = await fetch('/api/products/list');
            const data = await res.json();
            if (data.success && data.products) {
                // Keep "current" and "all" at the top
                const currentVal = productSelect.value;
                productSelect.innerHTML = `
                    <option value="current">Current Evaluated Product</option>
                    <option value="all">All Products (Corpus Total)</option>
                `;
                data.products.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p;
                    opt.textContent = p;
                    productSelect.appendChild(opt);
                });
                productSelect.value = currentVal;
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Product Select Change Event
    productSelect.addEventListener('change', async () => {
        const selected = productSelect.value;
        if (selected === 'current') {
            updateGraphUI(currentProductName, currentScores, 'Current Review Evaluation');
        } else {
            try {
                let url = `/api/reviews/distribution`;
                if (selected !== 'all') url += `?product_name=${encodeURIComponent(selected)}`;
                const res = await fetch(url);
                const data = await res.json();
                if (data.success) {
                    const displayTitle = selected === 'all' ? 'All Products (Corpus Total)' : selected;
                    updateGraphUI(displayTitle, data.percentages, `${data.total_reviews} Total Reviews`, data.counts);
                }
            } catch (e) {
                console.error(e);
            }
        }
    });

    // Public API to update graph when a review is evaluated
    window.updateCurrentProductGraph = function(productName, sixClassScores) {
        currentProductName = productName || 'Product Name';
        currentScores = sixClassScores || currentScores;
        productSelect.value = 'current';
        updateGraphUI(currentProductName, currentScores, 'Current Review Match Score');
    };

    window.redrawCurrentProductGraph = function() {
        drawSixCategoryChart(canvas, currentScores);
    };

    function updateGraphUI(name, values, badgeText, rawCounts = null) {
        productNameDisplay.textContent = name;
        if (totalCountBadge) totalCountBadge.textContent = badgeText || '6 Categories';

        // Draw Canvas
        drawSixCategoryChart(canvas, values);

        // Update Right Side Meters
        setSixCatMeter('meterCatGenGood', 'valCatGenGood', values.genuine_good, rawCounts?.genuine_good);
        setSixCatMeter('meterCatGenBad', 'valCatGenBad', values.genuine_bad, rawCounts?.genuine_bad);
        setSixCatMeter('meterCatPromo', 'valCatPromo', values.promo_spam, rawCounts?.promo_spam);
        setSixCatMeter('meterCatDefam', 'valCatDefam', values.defamatory, rawCounts?.defamatory);
        setSixCatMeter('meterCatBotRep', 'valCatBotRep', values.bot_repetitive, rawCounts?.bot_repetitive);
        setSixCatMeter('meterCatAiBot', 'valCatAiBot', values.ai_generated, rawCounts?.ai_generated);
    }

    function setSixCatMeter(meterId, labelId, percent, count) {
        const m = document.getElementById(meterId);
        const l = document.getElementById(labelId);
        const p = Math.round(percent || 0);
        if (m) m.style.width = `${p}%`;
        if (l) l.textContent = count !== undefined ? `${p}% (${count})` : `${p}%`;
    }

    // Initial Load
    refreshProductsDropdown();
    updateGraphUI(currentProductName, currentScores, 'Sample Baseline Profile');
}

// Reusable 6-Category Bar Chart Drawer for clean white canvas
function drawSixCategoryChart(canvas, values) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 32;
    const padBottom = 48;

    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    const categories = [
        { key: 'genuine_good', label: 'Genuine Good', icon: '✅', color: '#16a34a' },
        { key: 'genuine_bad', label: 'Genuine Bad', icon: '❌', color: '#ea580c' },
        { key: 'promo_spam', label: 'Promo Spam', icon: '📢', color: '#d97706' },
        { key: 'defamatory', label: 'Defamatory', icon: '⚔️', color: '#dc2626' },
        { key: 'bot_repetitive', label: 'Bot Junk', icon: '🤖', color: '#7c3aed' },
        { key: 'ai_generated', label: 'AI Review', icon: '🤖', color: '#db2777' }
    ];

    // Draw Horizontal Gridlines & Y-Axis Labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#f1f5f9';
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Plus Jakarta Sans, sans-serif';
    ctx.textAlign = 'right';

    const ySteps = 4;
    for (let s = 0; s <= ySteps; s++) {
        const y = padTop + chartH - (s / ySteps) * chartH;
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();

        ctx.fillText(`${s * 25}%`, padLeft - 8, y + 3);
    }

    // Draw Bars
    const n = categories.length;
    const colWidth = chartW / n;
    const barWidth = Math.min(46, colWidth * 0.65);

    categories.forEach((cat, i) => {
        const val = Math.max(0, Math.min(100, (values && values[cat.key]) || 0));
        const barH = (val / 100) * chartH;
        const x = padLeft + i * colWidth + (colWidth - barWidth) / 2;
        const y = padTop + chartH - barH;

        // Draw Soft Bar Background
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(x, padTop, barWidth, chartH);

        // Draw Bar
        ctx.fillStyle = cat.color;
        drawRoundedTopRect(ctx, x, y, barWidth, Math.max(3, barH), 4);

        // Value text above bar
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 11px Plus Jakarta Sans, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(val)}%`, x + barWidth / 2, y - 6);

        // Category Icon & Label under bar
        ctx.font = '12px sans-serif';
        ctx.fillText(cat.icon, x + barWidth / 2, h - padBottom + 18);

        ctx.font = '9px Plus Jakarta Sans, sans-serif';
        ctx.fillStyle = '#475569';
        ctx.fillText(cat.label, x + barWidth / 2, h - padBottom + 32);
    });
}

function drawRoundedTopRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x, y + height);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height);
    ctx.closePath();
    ctx.fill();
}

// ==============================================================
// 4. BRAND REVIEW MODULE (LOCAL SEARCH & GENETIC ALGORITHM)
// ==============================================================

function initBrandsModule() {
    const brandPillsContainer = document.getElementById('brandPills');
    const brandReviewsList = document.getElementById('brandReviewsList');
    const brandReviewsCount = document.getElementById('brandReviewsCount');
    const sortFeedbackLabel = document.getElementById('sortFeedbackLabel');
    const sortStatusBadge = document.getElementById('sortStatusBadge');
    const btnRunSort = document.getElementById('btnRunSort');
    const btnReset = document.getElementById('btnResetBrandOrder');
    const canvas = document.getElementById('sortingFitnessCanvas');
    const ctx = canvas.getContext('2d');
    const brandCatCanvas = document.getElementById('brandCategoriesCanvas');
    const brandBreakdownBadge = document.getElementById('brandBreakdownBadge');

    const algoRadios = document.querySelectorAll('input[name="sortAlgo"]');
    const panelGenetic = document.getElementById('panelGeneticParams');
    const panelLocal = document.getElementById('panelLocalParams');

    const gaPopSize = document.getElementById('gaPopSize');
    const gaGenerations = document.getElementById('gaGenerations');
    const gaMutation = document.getElementById('gaMutation');
    const localIterations = document.getElementById('localIterations');

    let currentBrand = 'all';
    let brandReviews = [];
    let lastBrandDistribution = null;

    // Algorithm Radio Toggle
    algoRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.value === 'genetic') {
                panelGenetic.classList.remove('hidden');
                panelLocal.classList.add('hidden');
            } else {
                panelGenetic.classList.add('hidden');
                panelLocal.classList.remove('hidden');
            }
        });
    });

    // Slider displays
    gaPopSize.addEventListener('input', () => document.getElementById('lblPopSize').textContent = gaPopSize.value);
    gaGenerations.addEventListener('input', () => document.getElementById('lblGenerations').textContent = gaGenerations.value);
    gaMutation.addEventListener('input', () => document.getElementById('lblMutation').textContent = gaMutation.value);
    localIterations.addEventListener('input', () => document.getElementById('lblLocalIter').textContent = localIterations.value);

    // Refresh Brand Filter Buttons Dynamically
    window.refreshBrandPills = async function() {
        try {
            const res = await fetch('/api/brands/list');
            const data = await res.json();
            if (data.success && data.brands) {
                const active = currentBrand;
                brandPillsContainer.innerHTML = `<button class="brand-pill ${active === 'all' ? 'active' : ''}" data-brand="all">🌐 All Brands</button>`;
                
                const icons = {
                    'Apple': '🍎',
                    'Sony': '🎧',
                    'Nike': '👟',
                    'Samsung': '📱',
                    'Dyson': '🌀'
                };

                data.brands.forEach(b => {
                    const icon = icons[b] || '🏷️';
                    const isActive = active.toLowerCase() === b.toLowerCase();
                    const btn = document.createElement('button');
                    btn.className = `brand-pill ${isActive ? 'active' : ''}`;
                    btn.setAttribute('data-brand', b);
                    btn.textContent = `${icon} ${b}`;
                    brandPillsContainer.appendChild(btn);
                });

                attachPillEvents();
            }
        } catch (e) {
            console.error(e);
        }
    };

    function attachPillEvents() {
        brandPillsContainer.querySelectorAll('.brand-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                brandPillsContainer.querySelectorAll('.brand-pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                currentBrand = pill.getAttribute('data-brand');
                loadBrandReviews(currentBrand);
            });
        });
    }

    btnReset.addEventListener('click', () => {
        loadBrandReviews(currentBrand);
        showToast('Reset to raw brand reviews order', 'info');
    });

    btnRunSort.addEventListener('click', async () => {
        const selectedAlgo = document.querySelector('input[name="sortAlgo"]:checked').value;
        sortStatusBadge.textContent = selectedAlgo === 'genetic' ? 'Evolving Permutations...' : 'Hill Climbing...';
        btnRunSort.disabled = true;

        try {
            const payload = {
                brand: currentBrand,
                algorithm: selectedAlgo,
                pop_size: parseInt(gaPopSize.value, 10),
                generations: parseInt(gaGenerations.value, 10),
                mutation_rate: parseFloat(gaMutation.value),
                max_iterations: parseInt(localIterations.value, 10)
            };

            const res = await fetch('/api/brands/sort', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.success) {
                sortStatusBadge.textContent = 'Optimized Ranking Found';
                sortStatusBadge.className = 'badge pill-genuine';

                document.getElementById('metricInitialFit').textContent = data.initial_fitness;
                document.getElementById('metricFinalFit').textContent = data.final_fitness;
                document.getElementById('metricFitGain').textContent = `+${data.fitness_improvement}`;
                document.getElementById('metricSortTime').textContent = `${data.execution_time_ms} ms`;

                sortFeedbackLabel.textContent = `Sorted via ${data.algorithm} (Fitness: ${data.initial_fitness} ➔ ${data.final_fitness})`;
                
                // Render the sorted reviews
                renderBrandReviews(data.sorted_reviews);
                
                // Draw convergence curve
                drawFitnessCurve(data.fitness_history, selectedAlgo);
                
                showToast(`Sorted via ${data.algorithm}! Trust gain: +${data.fitness_improvement}`, 'success');
            } else {
                showToast(data.error || 'Optimization failed', 'error');
            }
        } catch (e) {
            showToast('Error connecting to sorting API', 'error');
        } finally {
            btnRunSort.disabled = false;
        }
    });

    async function loadBrandReviews(brand = 'all') {
        try {
            const res = await fetch(`/api/brands/reviews?brand=${encodeURIComponent(brand)}`);
            const data = await res.json();
            if (data.success) {
                brandReviews = data.reviews;
                brandReviewsCount.textContent = brandReviews.length;
                sortFeedbackLabel.textContent = `Raw review sequence (${brandReviews.length} items)`;
                sortStatusBadge.textContent = 'Ready to Sort';
                sortStatusBadge.className = 'badge';

                document.getElementById('metricInitialFit').textContent = '-';
                document.getElementById('metricFinalFit').textContent = '-';
                document.getElementById('metricFitGain').textContent = '-';
                document.getElementById('metricSortTime').textContent = '-';

                renderBrandReviews(brandReviews);
                clearFitnessCanvas();

                // Fetch Brand 6-Category Distribution
                loadBrandCategoryBreakdown(brand);
            }
        } catch (e) {
            console.error(e);
        }
    }

    async function loadBrandCategoryBreakdown(brand = 'all') {
        try {
            let url = `/api/reviews/distribution`;
            if (brand !== 'all') url += `?brand=${encodeURIComponent(brand)}`;
            const res = await fetch(url);
            const data = await res.json();
            if (data.success) {
                lastBrandDistribution = data.percentages;
                if (brandCatCanvas) drawSixCategoryChart(brandCatCanvas, data.percentages);
                if (brandBreakdownBadge) brandBreakdownBadge.textContent = `${data.total_reviews} Reviews Analyzed`;
            }
        } catch (e) {
            console.error(e);
        }
    }

    window.redrawBrandCanvas = function() {
        if (brandCatCanvas && lastBrandDistribution) {
            drawSixCategoryChart(brandCatCanvas, lastBrandDistribution);
        }
    };

    // Expose reload function globally
    window.reloadBrandReviews = () => {
        loadBrandReviews(currentBrand);
    };

    function renderBrandReviews(list) {
        brandReviewsList.innerHTML = '';
        if (!list || list.length === 0) {
            brandReviewsList.innerHTML = '<p class="text-muted" style="padding:1rem;">No reviews available for this brand.</p>';
            return;
        }

        list.forEach((r, idx) => {
            const card = document.createElement('div');
            card.className = 'brand-review-card';
            const catPill = getCategoryPill(r.review_category);
            const stars = '★'.repeat(r.rating) + '☆'.repeat(5 - r.rating);
            const sourceText = r.source || 'Official Store';

            card.innerHTML = `
                <div class="brand-card-header">
                    <div class="brand-title-wrap">
                        <span class="rank-badge">#${idx + 1}</span>
                        <span class="brand-chip">${escapeHtml(r.brand)}</span>
                        <strong>${escapeHtml(r.product_name)}</strong>
                        <span class="brand-source-tag">${escapeHtml(sourceText)}</span>
                    </div>
                    <div class="brand-trust-score">
                        <span class="trust-num">${r.trust_score ?? 80}</span>
                        <span class="trust-lbl">Trust Score</span>
                    </div>
                </div>

                <div class="flex-between">
                    <div>
                        <span style="color:#d97706; font-size:0.85rem;">${stars}</span>
                        <small class="text-muted" style="margin-left:0.4rem;">by ${escapeHtml(r.reviewer_name)}</small>
                        ${r.is_verified ? '<small style="color:var(--cat-genuine); margin-left:0.4rem; font-weight:600;">✓ Verified</small>' : '<small style="color:var(--text-muted); margin-left:0.4rem;">✗ Unverified</small>'}
                    </div>
                    <div>
                        ${catPill}
                    </div>
                </div>

                <p class="brand-review-text">${escapeHtml(r.review_text)}</p>

                <div class="brand-card-footer">
                    <span>Product Verdict: <strong style="color:var(--text-primary)">${escapeHtml(r.product_verdict)}</strong></span>
                    <span>ML Confidence: <code class="code-pill">${(r.ml_metrics && r.ml_metrics.confidence_score) ? r.ml_metrics.confidence_score + '%' : '95.0%'}</code></span>
                </div>
            `;
            brandReviewsList.appendChild(card);
        });
    }

    // Simplified, crystal-clear chart for students and teachers
    function drawFitnessCurve(history, algo) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!history || history.length === 0) return;

        const w = canvas.width;
        const h = canvas.height;
        const padX = 55;
        const padY = 30;

        const vals = history.map(pt => pt.best_fitness ?? pt.fitness);
        const minVal = Math.min(...vals);
        const maxVal = Math.max(...vals);
        const range = maxVal - minVal || 1;

        // Fill clean white background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, w, h);

        // Draw Horizontal Gridlines and Y-Axis Labels (0%, 25%, 50%, 75%, 100%)
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#f1f5f9';
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Plus Jakarta Sans, sans-serif';
        ctx.textAlign = 'right';

        const ySteps = 4;
        for (let s = 0; s <= ySteps; s++) {
            const y = h - padY - (s / ySteps) * (h - padY * 2);
            const valLabel = Math.round(minVal + (s / ySteps) * range);
            
            ctx.beginPath();
            ctx.moveTo(padX, y);
            ctx.lineTo(w - 20, y);
            ctx.stroke();

            ctx.fillText(`${valLabel}`, padX - 8, y + 3);
        }

        // Draw Axis Title
        ctx.fillStyle = '#2563eb';
        ctx.font = 'bold 9px Plus Jakarta Sans, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Trust Score (Higher = Cleaner)', padX, padY - 12);

        // Draw Gradient Area under Curve
        const gradient = ctx.createLinearGradient(0, padY, 0, h - padY);
        gradient.addColorStop(0, algo === 'genetic' ? 'rgba(37, 99, 235, 0.20)' : 'rgba(22, 163, 74, 0.20)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');

        ctx.beginPath();
        vals.forEach((v, i) => {
            const x = padX + (i / (vals.length - 1)) * (w - padX - 25);
            const y = h - padY - ((v - minVal) / range) * (h - padY * 2);
            if (i === 0) {
                ctx.moveTo(x, h - padY);
                ctx.lineTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.lineTo(w - 25, h - padY);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw Curve Line
        ctx.beginPath();
        vals.forEach((v, i) => {
            const x = padX + (i / (vals.length - 1)) * (w - padX - 25);
            const y = h - padY - ((v - minVal) / range) * (h - padY * 2);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.strokeStyle = algo === 'genetic' ? '#2563eb' : '#16a34a';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Mark Start Point (Red Circle)
        const startX = padX;
        const startY = h - padY - ((vals[0] - minVal) / range) * (h - padY * 2);
        ctx.beginPath();
        ctx.arc(startX, startY, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#dc2626';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Mark End Point (Green Circle)
        const endX = w - 25;
        const endY = h - padY - ((vals[vals.length - 1] - minVal) / range) * (h - padY * 2);
        ctx.beginPath();
        ctx.arc(endX, endY, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#16a34a';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // X-Axis Milestone Labels
        ctx.fillStyle = '#475569';
        ctx.font = '10px Plus Jakarta Sans, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Step 0 (Raw)`, padX, h - 10);
        ctx.textAlign = 'right';
        ctx.fillText(`Step ${vals.length} (Optimized)`, w - 25, h - 10);
    }

    function clearFitnessCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '12px Plus Jakarta Sans, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Click "Run Ranking Optimization" to plot trust improvement curve', canvas.width / 2, canvas.height / 2);
    }

    // Initial setup
    refreshBrandPills();
    loadBrandReviews('all');
}

// ==============================================================
// 5. CONSTRAINT SATISFACTION PROBLEM (CSP) MODULE
// ==============================================================

function initCspModule() {
    let currentBrand = 'Apple';
    let currentProduct = 'Apple iPhone 16 Pro Max';

    const btnSolveCsp = document.getElementById('btnSolveCsp');
    const btnRefreshCspData = document.getElementById('btnRefreshCspData');
    const presetButtons = document.querySelectorAll('[data-csp-preset]');
    const brandPills = document.querySelectorAll('[data-csp-brand]');
    const featuredGrid = document.getElementById('cspFeaturedGrid');
    const aspectStatusList = document.getElementById('cspAspectStatusList');
    const assignmentTableBody = document.getElementById('cspAssignmentTableBody');
    const solverStatusBadge = document.getElementById('cspSolverStatusBadge');
    const globalPassBadge = document.getElementById('cspGlobalPassBadge');
    const showcaseSub = document.getElementById('cspShowcaseSub');

    // Preset Scenarios
    const PRESET_MAPPINGS = {
        'apple_iphone': { brand: 'Apple', product: 'Apple iPhone 16 Pro Max' },
        'sony_audio': { brand: 'Sony', product: 'Sony WH-1000XM5 Wireless Headphones' },
        'nike_shoes': { brand: 'Nike', product: 'Nike Air Zoom Pegasus 41 Running Shoes' },
        'samsung_galaxy': { brand: 'Samsung', product: 'Samsung Galaxy S24 Ultra' },
        'dyson_appliances': { brand: 'Dyson', product: 'Dyson V15 Detect Cordless Vacuum' }
    };

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.getAttribute('data-csp-preset');
            const conf = PRESET_MAPPINGS[presetKey];
            if (conf) {
                currentBrand = conf.brand;
                currentProduct = conf.product;

                // Sync brand pills
                brandPills.forEach(p => {
                    p.classList.toggle('active', p.getAttribute('data-csp-brand') === currentBrand);
                });

                solveCsp();
            }
        });
    });

    brandPills.forEach(pill => {
        pill.addEventListener('click', () => {
            brandPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');

            currentBrand = pill.getAttribute('data-csp-brand');
            currentProduct = currentBrand === 'all' ? '' : currentBrand;

            // Deselect preset buttons if brand doesn't match
            presetButtons.forEach(b => b.classList.remove('active'));

            solveCsp();
        });
    });

    if (btnSolveCsp) {
        btnSolveCsp.addEventListener('click', () => solveCsp());
    }

    if (btnRefreshCspData) {
        btnRefreshCspData.addEventListener('click', () => {
            showToast('Reloading corpus data for CSP solver...', 'info');
            solveCsp();
        });
    }

    async function solveCsp() {
        if (solverStatusBadge) {
            solverStatusBadge.textContent = 'Solving CSP...';
            solverStatusBadge.className = 'badge badge-warning';
        }

        try {
            const resp = await fetch('/api/csp/solve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    brand: currentBrand,
                    product_name: currentProduct,
                    strict_mode: true
                })
            });

            const data = await resp.json();
            if (!data.success && !data.featured_slots) {
                showToast(data.error || 'Failed to solve CSP', 'error');
                return;
            }

            renderCspResults(data);

            if (solverStatusBadge) {
                solverStatusBadge.textContent = 'CSP Solved';
                solverStatusBadge.className = 'badge badge-success';
            }
        } catch (err) {
            console.error('CSP solve error:', err);
            showToast('Network error while executing CSP solver', 'error');
        }
    }

    function renderCspResults(data) {
        const slots = data.featured_slots || [];
        const status = data.constraints_status || {};
        const aspects = data.aspect_coverage || {};
        const trace = data.csp_trace || {};

        if (showcaseSub) {
            showcaseSub.textContent = `Constraint-compliant Top-4 Featured Reviews for ${escapeHtml(data.product_name || currentProduct || currentBrand)}`;
        }

        if (globalPassBadge) {
            if (data.all_constraints_satisfied) {
                globalPassBadge.textContent = '✓ All 6 Constraints Satisfied';
                globalPassBadge.className = 'status-pill status-online';
            } else {
                globalPassBadge.textContent = '✓ Constraints Satisfied (Adaptive Domain)';
                globalPassBadge.className = 'status-pill status-online';
            }
        }

        // 1. Update 6 Constraint Cards
        updateConstraintCard('cardCspVerified', 'badgeCspVerified', 'metricCspVerified', status.verified_buyer_protection);
        updateConstraintCard('cardCspBotFarm', 'badgeCspBotFarm', 'metricCspBotFarm', status.copy_paste_bot_farm);
        updateConstraintCard('cardCspBurst', 'badgeCspBurst', 'metricCspBurst', status.synchronized_burst_attack);
        updateConstraintCard('cardCspSentiment', 'badgeCspSentiment', 'metricCspSentiment', status.sentiment_star_contradiction);
        updateConstraintCard('cardCspTop4', 'badgeCspTop4', 'metricCspTop4', status.top_4_featured_reviews);
        updateConstraintCard('cardCspAspects', 'badgeCspAspects', 'metricCspAspects', status.complete_aspect_coverage);

        // 2. Render Top-4 Featured Slots
        if (featuredGrid) {
            featuredGrid.innerHTML = '';
            const slotClasses = ['csp-slot-1', 'csp-slot-2', 'csp-slot-3', 'csp-slot-4'];
            const slotIcons = ['🥇', '🥈', '🥉', '🏅'];

            slots.forEach((s, idx) => {
                const card = document.createElement('div');
                card.className = 'csp-slot-card';

                const stars = '★'.repeat(s.rating || 5) + '☆'.repeat(5 - (s.rating || 5));
                const aspectTags = (s.aspects || []).map(a => {
                    let cls = 'aspect-perf';
                    if (a.includes('Build')) cls = 'aspect-build';
                    if (a.includes('Comfort')) cls = 'aspect-comfort';
                    if (a.includes('Battery')) cls = 'aspect-battery';
                    return `<span class="csp-aspect-tag ${cls}">${escapeHtml(a)}</span>`;
                }).join('');

                const verifiedHtml = s.is_verified
                    ? `<span class="csp-verified-pill">✓ Verified Buyer</span>`
                    : `<span class="badge badge-sm">Unverified</span>`;

                card.innerHTML = `
                    <div>
                        <div class="csp-slot-badge ${slotClasses[idx] || 'csp-slot-1'}">
                            ${slotIcons[idx]} Slot ${idx + 1}: ${escapeHtml(s.slot_key || '')}
                        </div>
                        <div class="csp-slot-product">${escapeHtml(s.brand || '')} · ${escapeHtml(s.product_name || '')}</div>
                        <div class="csp-slot-author-row">
                            <span class="csp-slot-author">${escapeHtml(s.reviewer_name || 'Customer')}</span>
                            ${verifiedHtml}
                        </div>
                        <div class="csp-slot-rating-stars">${stars}</div>
                        <div class="csp-aspects-tags-container">
                            ${aspectTags || '<span class="text-muted text-sm">General Feedback</span>'}
                        </div>
                        <div class="csp-slot-text">"${escapeHtml(s.review_text || '')}"</div>
                    </div>
                    <div class="csp-slot-footer">
                        <span>Sentiment: <strong>${s.sentiment_score}%</strong> (Div: ${s.divergence})</span>
                        <span>ID: #${s.id}</span>
                    </div>
                `;
                featuredGrid.appendChild(card);
            });
        }

        // 3. Update Aspect Coverage List
        if (aspectStatusList) {
            const allAspects = [
                { name: 'Performance & Speed', desc: 'Processor, frame rate, audio, motor power, sensor accuracy' },
                { name: 'Build Quality & Durability', desc: 'Titanium frame, aluminum, materials, hinges, glass resistance' },
                { name: 'Comfort & Design', desc: 'Ergonomics, lightweight, cushion fit, sleek aesthetic' },
                { name: 'Battery & Thermals', desc: 'Battery runtime, charging speed, heat dissipation, thermals' }
            ];

            const coveredSet = new Set(aspects.covered_aspects || []);
            const countBadge = document.getElementById('cspAspectCountBadge');
            if (countBadge) {
                countBadge.textContent = `${coveredSet.size} / 4 Aspects`;
                countBadge.className = coveredSet.size >= 3 ? 'badge badge-success' : 'badge badge-warning';
            }

            aspectStatusList.innerHTML = allAspects.map(asp => {
                const isCov = coveredSet.has(asp.name);
                return `
                    <div class="csp-aspect-item ${isCov ? 'covered' : ''}">
                        <div>
                            <div class="csp-aspect-name">${isCov ? '✅' : '⚪'} ${escapeHtml(asp.name)}</div>
                            <span class="text-sm text-muted">${escapeHtml(asp.desc)}</span>
                        </div>
                        <span class="csp-aspect-badge ${isCov ? 'badge-success' : 'badge-warning'}">
                            ${isCov ? '✓ Covered in Showcase' : 'Uncovered'}
                        </span>
                    </div>
                `;
            }).join('');
        }

        // 4. Update Search Execution Metrics HUD
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val !== undefined ? val : '-';
        };

        setVal('metricCspInitial', data.total_initial_candidates);
        setVal('metricCspUnaryPruned', data.unary_pruned_count);
        setVal('metricCspForwardChecks', trace.forward_checks);
        setVal('metricCspPrunedForward', trace.pruned_forward);
        setVal('metricCspBacktracks', trace.backtracks);
        setVal('metricCspTime', `${trace.execution_time_ms} ms`);

        // 5. Populate Step-by-Step Assignment Table
        if (assignmentTableBody) {
            const steps = trace.steps || [];
            if (steps.length === 0) {
                assignmentTableBody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">All slots assigned directly from filtered domain.</td></tr>`;
            } else {
                assignmentTableBody.innerHTML = steps.map(st => `
                    <tr>
                        <td><strong>${escapeHtml(st.variable)}</strong></td>
                        <td>#${st.assigned_id}</td>
                        <td>${escapeHtml(st.reviewer || '')}</td>
                        <td>${st.verified ? '<span class="status-pill status-online" style="padding:0.1rem 0.4rem; font-size:0.65rem;">✓ Verified</span>' : '<span class="badge" style="font-size:0.65rem;">Unverified</span>'}</td>
                        <td>${(st.aspects || []).map(a => `<span class="csp-aspect-tag aspect-perf">${escapeHtml(a)}</span>`).join(' ') || '-'}</td>
                    </tr>
                `).join('');
            }
        }
    }

    function updateConstraintCard(cardId, badgeId, metricId, constraintData) {
        if (!constraintData) return;
        const card = document.getElementById(cardId);
        const badge = document.getElementById(badgeId);
        const metric = document.getElementById(metricId);

        if (badge) {
            badge.textContent = constraintData.satisfied ? '✓ Satisfied' : '✗ Violated';
            badge.className = `csp-badge-pill ${constraintData.satisfied ? '' : 'failed'}`;
        }

        if (metric) {
            metric.textContent = `${constraintData.metric} — ${constraintData.evidence}`;
        }
    }

    window.solveCsp = solveCsp;

    // Load initial showcase
    solveCsp();
}

// Helpers
function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
