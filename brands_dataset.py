from ml_classifier import classify_review_with_ml
from sorting_algorithms import calculate_review_trust_score

RAW_BRAND_REVIEWS = [
    # =========================================================================
    # 1. APPLE (Official Store Reviews & Real Customer Feedback)
    # =========================================================================
    {
        "id": 101,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "Marcus Vance",
        "rating": 5,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "The 5x tetraprism telephoto lens and battery life upgrade are substantial improvements over my 14 Pro. Action button and camera control button take a day to get used to, but build quality and titanium frame are top notch."
    },
    {
        "id": 102,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "iFanatic_Deals",
        "rating": 5,
        "is_verified": False,
        "source": "Third-Party Reseller",
        "review_text": "AMAZING PRODUCT BEST EVER IN THE WORLD! BUY NOW GUYS GUARANTEED 100% SATISFACTION MIRACLE PHONE! CLICK HERE USE PROMO CODE APPLE20 BUY TODAY FAST!!"
    },
    {
        "id": 103,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "AndroidFanboy99",
        "rating": 1,
        "is_verified": False,
        "source": "Competitor Smear Campaign",
        "review_text": "BUY FROM SAMSUNG INSTEAD! APPLE IS A TOTAL SCAM COMPANY STEALING CUSTOMER MONEY WITH COMPLETE FRAUD! WORST TRASH EVER MADE BOYCOTT THIS JUNK!!"
    },
    {
        "id": 104,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "bot_apple_01",
        "rating": 5,
        "is_verified": False,
        "source": "Bot Farm Network",
        "review_text": "Good good good good good good good good good good good good good."
    },
    {
        "id": 105,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "Clara Higgins",
        "rating": 2,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "Camera control capacitive button is finicky with cases on. Device overheats when fast wireless charging on long commutes. For a $1200 phone, expected smoother thermals."
    },
    {
        "id": 106,
        "brand": "Apple",
        "product_name": "Apple MacBook Pro 16\" M3 Max",
        "reviewer_name": "Dev_Architect_Leo",
        "rating": 5,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "The M3 Max renders 8K ProRes timelines without dropping a single frame. Battery lasts a full 14-hour workday of running Docker containers and compiling Rust code. The Liquid Retina XDR screen is color accurate out of the box."
    },
    {
        "id": 107,
        "brand": "Apple",
        "product_name": "Apple Watch Ultra 2",
        "reviewer_name": "Triathlete_Dave",
        "rating": 3,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "Action button is great for transition intervals during triathlons. GPS tracks accurately under heavy tree canopies. However, battery life is only 36 hours compared to Garmin's two weeks."
    },
    {
        "id": 108,
        "brand": "Apple",
        "product_name": "Apple AirPods Pro 2",
        "reviewer_name": "SyntheticBot_User",
        "rating": 5,
        "is_verified": False,
        "source": "AI-Generated Astroturfing",
        "review_text": "In conclusion, the Apple AirPods Pro 2 is a true testament to modern engineering. It stands out as a game-changer with seamless integration and stellar audio quality. Furthermore, whether you are a commuter or an athlete, this product boasts an impressive noise cancellation. Overall, I highly recommend this incredible piece of technology. A must-have addition to any routine!"
    },
    {
        "id": 109,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "Elena Rostova",
        "rating": 5,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "The A18 Pro chip renders 4K ProRes 120fps video smoothly without stuttering. The Super Retina XDR OLED display has thinner bezels and exceptional peak brightness under direct sunlight."
    },
    {
        "id": 110,
        "brand": "Apple",
        "product_name": "Apple iPhone 16 Pro Max",
        "reviewer_name": "Jordan K.",
        "rating": 4,
        "is_verified": True,
        "source": "Apple Store Official (apple.com)",
        "review_text": "Comfortable ergonomic curve along the grade 5 titanium frame feels noticeably lighter in hands. Fast charging reaches 50% in 30 minutes, and battery easily lasts 2 days of moderate use."
    },

    # =========================================================================
    # 2. SONY (Official Sony Store & Verified Feedback)
    # =========================================================================
    {
        "id": 201,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "AudioEnthusiast",
        "rating": 5,
        "is_verified": True,
        "source": "Sony Official Store",
        "review_text": "Active noise cancellation creates complete silence in open office spaces. LDAC support brings out crisp instrument separation on lossless tracks. Earcups do get warm after 4 hours, but fold mechanism is sleek."
    },
    {
        "id": 202,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "Bose_Loyalist",
        "rating": 1,
        "is_verified": False,
        "source": "Competitor Smear Campaign",
        "review_text": "BUY FROM BOSE INSTEAD! SONY HEADPHONES ARE A COMPLETE FRAUD AND SCAM COMPANY STEALING MONEY! BROKEN CRAP GARBAGE BOYCOTT NOW!!"
    },
    {
        "id": 203,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "PromoBlast_Audio",
        "rating": 5,
        "is_verified": False,
        "source": "Third-Party Coupon Blast",
        "review_text": "AMAZING PRODUCT BEST EVER MIRACLE SOUND! 100% GUARANTEED SATISFACTION MUST BUY TODAY WITH DISCOUNT LINK FAST!!"
    },
    {
        "id": 204,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "Jordan P.",
        "rating": 3,
        "is_verified": True,
        "source": "Sony Official Store",
        "review_text": "Sound is great, but the new headband hinge doesn't fold flat into a compact pouch like the XM4 did. Takes up way more space in my backpack."
    },
    {
        "id": 205,
        "brand": "Sony",
        "product_name": "Sony PlayStation 5 Pro Console",
        "reviewer_name": "Gamer_Vortex",
        "rating": 5,
        "is_verified": True,
        "source": "PlayStation Direct Official",
        "review_text": "PSSR AI upscaling produces remarkably sharp 4K 60fps image quality in Spider-Man 2 and Final Fantasy VII Rebirth with ray tracing enabled. The 2TB SSD is a welcome expansion over the base PS5."
    },
    {
        "id": 206,
        "brand": "Sony",
        "product_name": "Sony Bravia XR OLED 65\" 4K TV",
        "reviewer_name": "CinemaLover_Kev",
        "rating": 5,
        "is_verified": True,
        "source": "Sony Official Store",
        "review_text": "Cognitive Processor XR produces inky pure blacks and striking peak specular highlights for HDR movies. Acoustic Surface Audio+ vibrates the glass panel to project dialogue directly from actors' mouths."
    },
    {
        "id": 207,
        "brand": "Sony",
        "product_name": "Sony WF-1000XM5 Wireless Earbuds",
        "reviewer_name": "AI_Generated_Reviewer",
        "rating": 5,
        "is_verified": False,
        "source": "AI-Generated Synthetic Review",
        "review_text": "In conclusion, these Sony earbuds are a true testament to cutting-edge audio innovation. It stands out as a game-changer with sleek and stylish ergonomics and stellar noise cancellation. Delving into sound quality, it boasts an impressive dynamic range. Without a shadow of a doubt, a worthy investment."
    },
    {
        "id": 208,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "Liam Vance",
        "rating": 5,
        "is_verified": True,
        "source": "Sony Official Store",
        "review_text": "Dual noise sensor microphones block engine roar on transatlantic flights. Multipoint Bluetooth connectivity switches between laptop and phone seamlessly."
    },
    {
        "id": 209,
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5 Wireless Headphones",
        "reviewer_name": "Sarah Lin",
        "rating": 4,
        "is_verified": True,
        "source": "Sony Official Store",
        "review_text": "Plush synthetic leather earcups provide cloud-like comfort for 8-hour mixing sessions. 30-hour battery life with USB-C quick charge is reliable."
    },

    # =========================================================================
    # 3. NIKE (Nike.com Official Store & Athletes Feedback)
    # =========================================================================
    {
        "id": 301,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "MarathonRunner_Dan",
        "rating": 5,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "ReactX foam gives noticeably higher energy return compared to version 40. Midfoot lockdown is secure without causing blisters during 15-mile long runs. True to size with a breathable engineered mesh upper."
    },
    {
        "id": 302,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "SneakerBot_Alpha",
        "rating": 5,
        "is_verified": False,
        "source": "Bot Farm Network",
        "review_text": "Shoe shoe shoe shoe shoe shoe shoe shoe shoe shoe shoe."
    },
    {
        "id": 303,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "Hoka_Fanatic",
        "rating": 1,
        "is_verified": False,
        "source": "Competitor Smear Campaign",
        "review_text": "BUY FROM HOKA OR BROOKS INSTEAD! NIKE IS AN OVERPRICED SCAM COMPANY STEALING YOUR KNEES WITH COMPLETE FRAUD CUSHIONING! WORST TRASH EVER!!"
    },
    {
        "id": 304,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "Samantha Lee",
        "rating": 4,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "Very comfortable daily trainer for 5K workouts. Forefoot is slightly narrow if you have wide feet, so consider ordering half a size up."
    },
    {
        "id": 305,
        "brand": "Nike",
        "product_name": "Nike Vaporfly 3 Road Racing Shoes",
        "reviewer_name": "Sub3_Pacer_Jim",
        "rating": 5,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "Ran a personal best of 2:54 in the Boston Marathon with these. The full-length carbon Flyplate and ZoomX foam provide unmatched propulsion on downhills. Durability is limited to around 200 miles, but for race day it has no equal."
    },
    {
        "id": 306,
        "brand": "Nike",
        "product_name": "Nike Metcon 9 Training Shoes",
        "reviewer_name": "CrossFit_Coach_Mike",
        "rating": 5,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "Large Hyperlift plate in the heel gives immense stability for 400lb squats and clean-and-jerks. Lace lock system prevents laces from snagging on barbell knurling. Stiff for running, but phenomenal for heavy lifting."
    },
    {
        "id": 307,
        "brand": "Nike",
        "product_name": "Nike Air Force 1 '07",
        "reviewer_name": "AI_Fashion_Critic",
        "rating": 5,
        "is_verified": False,
        "source": "AI-Generated Synthetic Review",
        "review_text": "Overall, the Nike Air Force 1 stands out as a quintessential blend of comfort and timeless style. Delving into the craftsmanship, it boasts an impressive leather build that ticks all the right boxes. In summary, whether you are dressing up or down, this shoe is a must-have addition to any sneaker rotation."
    },
    {
        "id": 308,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "Coach_Derek",
        "rating": 5,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "ReactX foam under the heel and forefoot delivers lively energy return on tempo runs. Waffle-inspired rubber outsole provides durable grip on wet asphalt."
    },
    {
        "id": 309,
        "brand": "Nike",
        "product_name": "Nike Air Zoom Pegasus 41 Running Shoes",
        "reviewer_name": "Mia Chen",
        "rating": 4,
        "is_verified": True,
        "source": "Nike.com Official Store",
        "review_text": "Padded tongue and plush collar eliminate heel slip. Breathable mesh keeps feet cool during hot summer half-marathon training."
    },

    # =========================================================================
    # 4. SAMSUNG (Samsung.com Official Store & Verified Users)
    # =========================================================================
    {
        "id": 401,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy S24 Ultra",
        "reviewer_name": "TechReviewer_Neil",
        "rating": 5,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "The flat anti-reflective display is a game changer in bright sunlight. S-Pen integration remains unmatched for digital signatures and notes. Battery comfortably lasts 1.5 days on mixed 5G and Wi-Fi use."
    },
    {
        "id": 402,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy S24 Ultra",
        "reviewer_name": "DiscountFinder24",
        "rating": 5,
        "is_verified": False,
        "source": "Third-Party Affiliate Spam",
        "review_text": "AMAZING PRODUCT BEST PHONE EVER ON THE PLANET! BUY NOW FAST 100% SATISFACTION GUARANTEED MIRACLE DISPLAY! CLICK LINK NOW!!"
    },
    {
        "id": 403,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy S24 Ultra",
        "reviewer_name": "Brian Cole",
        "rating": 2,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "Vivid color profile mode is washed out compared to older Galaxy flagships. Sharp corners dig into palms without a bulky protective case."
    },
    {
        "id": 404,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy Z Fold 6",
        "reviewer_name": "MobileExecutive_Ron",
        "rating": 5,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "The square corners and thinner hinge make the cover screen much more comfortable for one-handed typing. Multitasking with three split apps while reviewing spreadsheets on the inner 7.6-inch AMOLED display is unmatched."
    },
    {
        "id": 405,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy Watch Ultra",
        "reviewer_name": "TrailRunner_Sam",
        "rating": 4,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "Dual-frequency GPS accurately tracked my open-water ocean swim and mountain trail hike. Titanium cushion case takes rock scuffs easily without scratching. The Quick Button is convenient for pause and lap markers."
    },
    {
        "id": 406,
        "brand": "Samsung",
        "product_name": "Samsung Neo QLED 8K Smart TV",
        "reviewer_name": "Synthetic_Home_Critic",
        "rating": 5,
        "is_verified": False,
        "source": "AI-Generated Synthetic Review",
        "review_text": "Upon unboxing, I was pleasantly surprised by the sleek and elegant bezel-less design. Delving into picture quality, the NQ8 AI Gen3 processor boasts an impressive upscaling prowess that raises the bar. In conclusion, whether for gaming or cinema, it delivers on all fronts without a shadow of a doubt."
    },
    {
        "id": 407,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy S24 Ultra",
        "reviewer_name": "TechArchitect_Raj",
        "rating": 5,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "Snapdragon 8 Gen 3 delivers blistering processing speed for ray-traced gaming. Quad-camera system with 50MP 5x periscope optical zoom captures crisp details."
    },
    {
        "id": 408,
        "brand": "Samsung",
        "product_name": "Samsung Galaxy S24 Ultra",
        "reviewer_name": "Claire Dupont",
        "rating": 5,
        "is_verified": True,
        "source": "Samsung.com Official Store",
        "review_text": "Titanium frame with satin finish feels premium and durable. 5000mAh battery effortlessly powers through 16 hours of heavy video streaming."
    },

    # =========================================================================
    # 5. DYSON (Dyson.com Official Store & Verified Customers)
    # =========================================================================
    {
        "id": 501,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "HomeMaker_Grace",
        "rating": 5,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "Green optical laser illumination reveals microscopic dust on hardwood floors that traditional vacuums completely miss. Piezo acoustic sensor automates suction power intelligently on thick carpets."
    },
    {
        "id": 502,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "Shark_Defender",
        "rating": 1,
        "is_verified": False,
        "source": "Competitor Smear Campaign",
        "review_text": "BUY FROM SHARK INSTEAD! DYSON IS A CRIMINAL SCAM COMPANY STEALING CONSUMER MONEY WITH COMPLETE PLASTIC FRAUD! BOYCOTT THIS EXPENSIVE TRASH!!"
    },
    {
        "id": 503,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "PromoRobot_Clean",
        "rating": 5,
        "is_verified": False,
        "source": "Third-Party Promo Spam",
        "review_text": "AMAZING PRODUCT BEST EVER MIRACLE CLEANER! 100% GUARANTEED BUY NOW WITH SPECIAL DISCOUNT CODE CLEAN50 FAST!!"
    },
    {
        "id": 504,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "Oliver Twist",
        "rating": 3,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "Cleans exceptionally well, but holding the trigger continuously for 30 minutes causes hand cramping. Would prefer a toggle power button."
    },
    {
        "id": 505,
        "brand": "Dyson",
        "product_name": "Dyson Airwrap Multi-Styler Complete",
        "reviewer_name": "SalonStylist_Chloe",
        "rating": 5,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "The Coanda airflow technology styles and curls hair without extreme heat damage. The smoothing dryer attachment eliminates frizz and flyaways in seconds. Expensive, but cuts morning blowout routine in half."
    },
    {
        "id": 506,
        "brand": "Dyson",
        "product_name": "Dyson Supersonic Nural Hair Dryer",
        "reviewer_name": "BeautyExpert_Hannah",
        "rating": 5,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "ToF sensor automatically reduces heat to 55°C when the dryer gets close to your scalp to protect root moisture. Capsule illumination changes color based on distance. Powerful airflow dries thick hair in 6 minutes."
    },
    {
        "id": 507,
        "brand": "Dyson",
        "product_name": "Dyson Purifier Hot+Cool Formaldehyde",
        "reviewer_name": "AI_Home_Reviewer",
        "rating": 5,
        "is_verified": False,
        "source": "AI-Generated Synthetic Review",
        "review_text": "In terms of air purification and thermal regulation, this unit is a true game-changer. Meticulously designed, it seamlessly circulates clean warm air throughout large living spaces. In conclusion, Dyson has delivered an impressive piece of engineering. A must-have for every modern home."
    },
    {
        "id": 508,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "CleanHome_Sarah",
        "rating": 5,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "Hyperdymium motor generates tremendous 240AW suction that extracts deep pet hair from high-pile rugs. Click-in battery provides a full 60 minutes of fade-free cleaning."
    },
    {
        "id": 509,
        "brand": "Dyson",
        "product_name": "Dyson V15 Detect Cordless Vacuum",
        "reviewer_name": "Dr_Arun_Patel",
        "rating": 4,
        "is_verified": True,
        "source": "Dyson.com Official Store",
        "review_text": "Whole-machine HEPA filtration traps 99.99% of microscopic allergens. Ergonomic wand is well-balanced for ceiling dust removal and cleans smoothly."
    }
]

# In-memory storage for user-submitted reviews routed from Tab 1
DYNAMIC_BRAND_REVIEWS = []

PROCESSED_BRAND_REVIEWS = None

def add_dynamic_brand_review(brand, product_name, reviewer_name, rating, review_text, is_verified=False, source="Submitted via Fake Review Detector"):
    """
    Dynamically registers a newly submitted review into the Brand Review corpus.
    Allows reviews typed in Tab 1 (Apple, Nike, etc.) to immediately appear in Tab 2.
    """
    global PROCESSED_BRAND_REVIEWS
    
    brand_clean = (brand or "General").strip().title()
    new_id = 1000 + len(DYNAMIC_BRAND_REVIEWS) + 1
    
    item_raw = {
        "id": new_id,
        "brand": brand_clean,
        "product_name": product_name.strip(),
        "reviewer_name": reviewer_name.strip() or "Customer",
        "rating": int(rating),
        "is_verified": bool(is_verified),
        "source": source,
        "review_text": review_text.strip()
    }

    ml_res = classify_review_with_ml(
        rating=item_raw["rating"],
        review_text=item_raw["review_text"],
        is_verified=item_raw["is_verified"],
        reviewer_name=item_raw["reviewer_name"]
    )

    item = dict(item_raw)
    item["review_category"] = ml_res["review_category"]
    item["product_verdict"] = ml_res["product_verdict"]
    item["product_quality_score"] = ml_res["product_quality_score"]
    item["product_quality_description"] = ml_res["product_quality_description"]
    item["ml_metrics"] = ml_res["ml_metrics"]
    item["feature_breakdown"] = ml_res["feature_breakdown"]
    item["trust_score"] = calculate_review_trust_score(item)

    DYNAMIC_BRAND_REVIEWS.append(item)
    # Invalidate cached list so next call recalculates
    PROCESSED_BRAND_REVIEWS = None
    return item

def get_processed_brand_reviews():
    """
    Returns the consolidated list of all brand reviews (Base official reviews + Dynamically added ones).
    Each review has its Supervised ML classification and DCG Trust Score precomputed.
    """
    global PROCESSED_BRAND_REVIEWS
    if PROCESSED_BRAND_REVIEWS is not None:
        return PROCESSED_BRAND_REVIEWS

    processed = []
    # 1. Process base official brand reviews
    for r in RAW_BRAND_REVIEWS:
        ml_res = classify_review_with_ml(
            rating=r["rating"],
            review_text=r["review_text"],
            is_verified=r["is_verified"],
            reviewer_name=r["reviewer_name"]
        )

        item = dict(r)
        item["review_category"] = ml_res["review_category"]
        item["product_verdict"] = ml_res["product_verdict"]
        item["product_quality_score"] = ml_res["product_quality_score"]
        item["product_quality_description"] = ml_res["product_quality_description"]
        item["ml_metrics"] = ml_res["ml_metrics"]
        item["feature_breakdown"] = ml_res["feature_breakdown"]
        item["trust_score"] = calculate_review_trust_score(item)
        processed.append(item)

    # 2. Append dynamically submitted reviews
    for d in DYNAMIC_BRAND_REVIEWS:
        processed.append(d)

    PROCESSED_BRAND_REVIEWS = processed
    return PROCESSED_BRAND_REVIEWS
