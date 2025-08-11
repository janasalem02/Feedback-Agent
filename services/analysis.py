from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import threading
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import rapidfuzz
import re
import pickle
import os
import hashlib
import difflib  

from pymongo import MongoClient
# Motor removed - using only pymongo for sync operations

from .feedback_points_extractor import (
    FeedbackPointsExtractor, 
    PointsAnalysisResult, 
    extract_points_with_llama, 
    classify_point_with_llama
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# MongoDB configuration
from core.config import settings

mongo_url = getattr(settings, 'mongo_url', 'mongodb://localhost:27017/')
db_name = getattr(settings, 'database_name', 'audio_db')
criteria_collection_name = getattr(settings, 'criteria_collection', 'client')
results_collection_name = getattr(settings, 'collection_name', 'audio_files')

# Optimized MongoDB connection pool (تحسين تجمع الاتصالات)
client = MongoClient(
    mongo_url,
    maxPoolSize=20,  # Reduced from 50 for efficiency
    minPoolSize=5,   # Reduced from 10 for better resource management
    maxIdleTimeMS=30000,
    connectTimeoutMS=5000,
    serverSelectionTimeoutMS=5000
)
db = client[db_name]
criteria_collection = db[criteria_collection_name]
results_collection = db[results_collection_name]

# Async MongoDB client for concurrent operations - تعطيل مؤقت
# async_client = AsyncIOMotorClient(
#     mongo_url,
#     maxPoolSize=20,
#     minPoolSize=5,
#     maxIdleTimeMS=30000,
#     connectTimeoutMS=5000,
#     serverSelectionTimeoutMS=5000
# MongoDB collections (sync only)
_criteria_cache = {}
_cache_timeout = 900  # Extended to 15 minutes for better cache efficiency
_cache_lock = threading.Lock()

# Async HTTP session management
_async_session = None
_session_lock = asyncio.Lock()

# Pre-compiled regex patterns for performance (تجميع الأنماط مسبقاً)
NEGATIVE_REGEX_PATTERNS = [
    re.compile(r"لم\s+يكن\s+\w*متوقع"),
    re.compile(r"لم\s+\w*\s*مقبول"),
    re.compile(r"غير\s+\w+"),
    re.compile(r"ما\s+\w+\s*متوقع"),
    re.compile(r"مش\s+\w+"),
    re.compile(r"لم\s+يعجب"),
    re.compile(r"ما\s+عجب"),
    re.compile(r"أبدا\s+ما\s+عجب"),
]

# Pre-processed sentiment patterns as sets for O(1) lookup
STRONG_POSITIVE_PATTERNS = {
    "كانت على الوقت", "على الوقت المحدد", "في الوقت المحدد", "في الموعد", "بالموعد",
    "سريع", "سريعة", "بسرعة", "توصيل سريع", "توصيل ممتاز", "توصيل جيد",
    "يجاني بالموعد", "وصل بالموعد", "وصل في الوقت", "التوصيل كان كويس",
    "التوصيل كان على الوقت", "خدمة التوصيل كانت على الوقت",
    "جودة عالية", "جودة ممتازة", "جودة كويسة", "جودة جيدة", "نوعية كويسة",
    "خدمة ممتازة", "خدمة كويسة", "خدمة جيدة", "تعامل مريح", "تعامل كويس",
    "تعامل ممتاز", "خدمة العملاء كانت كويسة", "الخدمة كانت حلوة",
    "تعاملهم مريح", "تعاملهم مناسب", "تعاملهم كويس",
    "كويس", "حلو", "ممتاز", "جيد", "رائع", "عالي", "مريح", "مناسب", "لذيذ",
    "أعجبني", "عجبني", "حبيته", "أنصح", "ينصح", "استمتعت", "راضي",
    # إضافات جديدة للسياق العربي المعقد
    "زاكي ورايق", "طعم زاكي", "ريحة رائعة", "عالم آخر", "عالم اخر", 
    "تفتح النفس", "تغطي المكان", "محافظين على الجودة", "نفس الجودة",
    "خياري الأول", "دائما خياري", "ليس أي قهوة", "ليس اي قهوة", "مش أي قهوة",
    "غيره الطعم", "بني العميد غيره", "يسعدهم", "مميز ومختلف"
}

STRONG_NEGATIVE_PATTERNS = {
    "لم يكن متوقعًا", "لم يكن متوقعا", "لم تكن متوقعة", "غير متوقع", "غير متوقعة",
    "مش متوقع", "ما كان متوقع", "لم يكن مقبولا", "لم يكن مقبولاً", "لم يكن مقبول",
    "لم تكن مقبولة", "مش مقبول", "غير مقبول", "لم يكن جيد", "لم يكن جيدًا",
    "لم تكن جيدة", "مش جيد", "غير جيد", "لم يعجبني", "ما عجبني", "مش حلو",
    "أبدا ما عجبني", "ما أعجبني أبداً", "مش عاجبني", "ما حبيته",
    "مخيب للآمال", "مخيب للامال", "محبط", "مزعج", "مقرف", "بشع", "فظيع",
    "مقلق", "مؤذي", "لا يستحق", "ما يستاهل", "مش يستاهل", "مضيعة وقت",
    "مضيعة فلوس", "خسارة", "رديء", "سيئ", "تالف", "معطل", "خربان",
    "بطيء", "متأخر", "غالي", "مرتفع", "باهظ"
}

EXTRA_POSITIVE_PATTERNS = {
    "فاق التوقعات", "أحسن من المتوقع", "كان متوقع وأكتر", "عجبني كتير",
    "رائع", "ممتاز", "جميل جداً", "حلو كتير", "مبهر", "استثنائي",
    "أعجبني جداً", "حبيته كتير", "مميز", "فريد", "لذيذ جداً"
}

STRONG_NEGATIVE_WORDS = {"سيئ", "رديء", "فاشل", "مقرف", "بشع", "محبط", "مزعج"}

# Priority keyword mapping for faster criteria matching
PRIORITY_KEYWORDS = {
    "توصيل": ["توصيل", "تسليم", "delivery"],
    "طعم": ["طعم", "طعام", "taste", "food"],
    "سعر": ["سعر", "ثمن", "price", "cost"],
    "خدمة": ["خدمة", "service", "customer"]
}

# Enhanced semantic matching cache
_semantic_cache = {}
_semantic_cache_file = "semantic_cache.pkl"
_semantic_lock = threading.Lock()

async def get_async_session():
    """Get or create async HTTP session with proper headers"""
    global _async_session
    async with _session_lock:
        if _async_session is None or _async_session.closed:
            timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
            _async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
    return _async_session

async def close_async_session():
    """Close async HTTP session"""
    global _async_session
    if _async_session and not _async_session.closed:
        await _async_session.close()
        _async_session = None

class SimpleSemanticMatcher:
    """Simple semantic matching using keyword extraction and caching"""
    
    def __init__(self):
        self.load_cache()
    
    def detect_sentiment_override(self, point: str) -> str:
        """Detect if a point has strong sentiment indicators that should override classification"""
        point_lower = point.lower().strip()
        
        # ✅ فحص محسن للسياق العربي المعقد - أولوية للأنماط الإيجابية المعقدة
        
        # PRIORITY 1: فحص العبارات الإيجابية المعقدة أولاً (للنص المقدم)
        complex_positive_patterns = [
            "ليس أي قهوة", "ليس اي قهوة", "مش أي قهوة", "مش اي قهوة",
            "غيره الطعم", "بني العميد غيره", "طعم زاكي ورايق", 
            "عالم آخر", "عالم اخر", "تفتح النفس", "تغطي المكان",
            "محافظين على الجودة", "محافظين على نفس الجودة", "نفس الجودة",
            "خياري الأول", "دائما خياري", "يسعدهم", "ريحة تشعر"
        ]
        
        for pattern in complex_positive_patterns:
            if pattern in point_lower:
                logger.info(f"🟢 COMPLEX POSITIVE DETECTED: '{pattern}' in '{point[:50]}...'")
                return "positive"
        
        # PRIORITY 2: فحص الأنماط السلبية القوية (أولوية عالية)
        strong_negative_patterns = [
            # Negative expectation patterns - CRITICAL for fixing misclassification
            "أبدا ما عجبني", "أبداً ما عجبني", "أبدا ما عجبنى", "ابدا ما عجبني",
            "لم يعجبني", "ما عجبني", "مش عاجبني", "ما حبيته", "لم يعجبني أبداً",
            "لم يكن متوقعًا", "لم يكن متوقعا", "لم تكن متوقعة", "غير متوقع", "غير متوقعة", 
            "مش متوقع", "ما كان متوقع", "لم يكن مقبولا", "لم يكن مقبولاً", "لم يكن مقبول", 
            "لم تكن مقبولة", "مش مقبول", "غير مقبول", "لم يكن جيد", "لم يكن جيدًا", 
            "لم تكن جيدة", "مش جيد", "غير جيد", "مش حلو",
            
            # Strong negative indicators
            "مخيب للآمال", "مخيب للامال", "محبط", "مزعج", "مقرف", "بشع", "فظيع", 
            "مقلق", "مؤذي", "لا يستحق", "ما يستاهل", "مش يستاهل", "مضيعة وقت",
            "مضيعة فلوس", "خسارة", "رديء", "سيئ", "تالف", "معطل", "خربان",
            "بطيء", "متأخر", "غالي", "مرتفع", "باهظ"
        ]
        
        # Check for strong negative patterns
        for pattern in strong_negative_patterns:
            if pattern in point_lower:
                logger.info(f"🔴 NEGATIVE DETECTED: '{pattern}' in '{point[:50]}...'")
                return "negative"
        
        # PRIORITY 3: فحص الأنماط الإيجابية العادية
        strong_positive_patterns = [
            # Delivery and timing positive
            "كانت على الوقت", "على الوقت المحدد", "في الوقت المحدد", "في الموعد", "بالموعد", 
            "سريع", "سريعة", "بسرعة", "توصيل سريع", "توصيل ممتاز", "توصيل جيد",
            "يجاني بالموعد", "وصل بالموعد", "وصل في الوقت", "التوصيل كان كويس",
            "التوصيل كان على الوقت", "خدمة التوصيل كانت على الوقت",
            
            # Quality positive  
            "جودة عالية", "جودة ممتازة", "جودة كويسة", "جودة جيدة", "نوعية كويسة",
            
            # Service positive
            "خدمة ممتازة", "خدمة كويسة", "خدمة جيدة", "تعامل مريح", "تعامل كويس", 
            "تعامل ممتاز", "خدمة العملاء كانت كويسة", "الخدمة كانت حلوة",
            "تعاملهم مريح", "تعاملهم مناسب", "تعاملهم كويس",
            
            # General positive
            "كويس", "حلو", "ممتاز", "جيد", "رائع", "عالي", "مريح", "مناسب", "لذيذ",
            "أعجبني", "عجبني", "حبيته", "أنصح", "ينصح", "استمتعت", "راضي",
            "زاكي", "رايق", "طعم زاكي", "ريحة رائعة"
        ]
        
        # Check for strong positive patterns
        for pattern in strong_positive_patterns:
            if pattern in point_lower:
                logger.info(f"🟢 POSITIVE DETECTED: '{pattern}' in '{point[:50]}...'")
                return "positive"
        
        # Additional regex patterns for complex negative expressions
        import re
        negative_regex_patterns = [
            r"أبدا\s+ما\s+عجب",          # "أبدا ما عجبني" - CRITICAL
            r"لم\s+يكن\s+\w*متوقع",      # "لم يكن متوقعًا" variations
            r"لم\s+\w*\s*مقبول",         # "لم يكن مقبول" variations  
            r"غير\s+\w+",                # "غير متوقع" variations
            r"ما\s+\w+\s*متوقع",         # "ما كان متوقع" variations
            r"مش\s+\w+",                 # "مش متوقع" variations
            r"لم\s+يعجب",                # "لم يعجبني" variations
            r"ما\s+عجب",                 # "ما عجبني" variations
        ]
        
        # Check negative regex patterns
        for pattern in negative_regex_patterns:
            if re.search(pattern, point_lower):
                return "negative"
        
        # Advanced negative word combinations
        negative_combinations = [
            ("طعم", "أبدا"), ("طعم", "ما"), ("طعم", "لم"), ("طعم", "مش"),
            ("طعم", "سيئ"), ("رائحة", "سيئ"), ("جودة", "ضعيف"), ("جودة", "سيئ"),
            ("خدمة", "سيئ"), ("سعر", "غالي"), ("سعر", "مرتفع"), ("سعر", "باهظ"),
        ]
        
        for word1, word2 in negative_combinations:
            if word1 in point_lower and word2 in point_lower:
                return "negative"
        
        # Advanced positive word combinations (only if no negative found)
        positive_combinations = [
            ("وقت", "محدد"), ("موعد", "مناسب"), ("توصيل", "سريع"), ("خدمة", "مريح"),
            ("تعامل", "مريح"), ("تعامل", "مناسب"), ("خدمة", "كويس"), ("جودة", "عالي"),
            ("على", "الوقت"), ("في", "الموعد"), ("بالموعد", "المناسب")
        ]
        
        for word1, word2 in positive_combinations:
            if word1 in point_lower and word2 in point_lower:
                return "positive"
        
        # Final check for explicit negative words
        strong_negative_words = ["سيئ", "رديء", "فاشل", "مقرف", "بشع", "محبط", "مزعج"]
        for word in strong_negative_words:
            if word in point_lower:
                return "negative"
        
        return "neutral"  # No strong sentiment detected
    
    def load_cache(self):
        """Load semantic cache from file"""
        global _semantic_cache
        try:
            if os.path.exists(_semantic_cache_file):
                with open(_semantic_cache_file, 'rb') as f:
                    _semantic_cache = pickle.load(f)
        except Exception:
            _semantic_cache = {}
    
    def save_cache(self):
        """Save semantic cache to file"""
        try:
            with open(_semantic_cache_file, 'wb') as f:
                pickle.dump(_semantic_cache, f)
        except Exception:
            pass
    
    def extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text"""
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned.split()
        
        # Filter meaningful words (length > 2)
        keywords = {word for word in words if len(word) > 2}
        return keywords
    
    def calculate_similarity(self, point: str, criteria_name: str) -> float:
        """Calculate enhanced similarity score between point and criteria with Arabic-specific optimizations"""
        cache_key = hashlib.md5(f"{point}|{criteria_name}".encode()).hexdigest()
        
        # Check cache first
        with _semantic_lock:
            if cache_key in _semantic_cache:
                return _semantic_cache[cache_key]
        
        # Extract keywords
        point_keywords = self.extract_keywords(point)
        criteria_keywords = self.extract_keywords(criteria_name)
        
        if not point_keywords or not criteria_keywords:
            score = 0.0
        else:
            # 1. Basic Jaccard similarity
            intersection = len(point_keywords.intersection(criteria_keywords))
            union = len(point_keywords.union(criteria_keywords))
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # 2. Substring matching bonus
            substring_bonus = 0.0
            if criteria_name.lower() in point.lower() or point.lower() in criteria_name.lower():
                substring_bonus = 0.4
            
            # 3. Partial word matches (important for Arabic)
            partial_bonus = 0.0
            for p_word in point_keywords:
                for c_word in criteria_keywords:
                    if len(p_word) > 3 and len(c_word) > 3:
                        if p_word in c_word or c_word in p_word:
                            partial_bonus += 0.1
                        # Arabic character similarity
                        elif abs(len(p_word) - len(c_word)) <= 2:
                            common_chars = set(p_word) & set(c_word)
                            char_similarity = len(common_chars) / max(len(p_word), len(c_word))
                            if char_similarity > 0.6:
                                partial_bonus += 0.05
            
            # 4. SMART difflib-based similarity (NEW!)
            difflib_score = self.calculate_difflib_similarity(point, criteria_name)
            
            # 5. Length-based adjustment
            length_factor = min(len(point_keywords), len(criteria_keywords)) / max(len(point_keywords), len(criteria_keywords))
            
            # Combine all scores with difflib getting significant weight
            score = (jaccard_score * 0.3) + (substring_bonus * 0.25) + (partial_bonus * 0.15) + (difflib_score * 0.25) + (length_factor * 0.05)
            score = min(score, 1.0)  # Cap at 1.0
        
        # Cache the result
        with _semantic_lock:
            _semantic_cache[cache_key] = score
            
            # Periodically save cache (every 50 new entries)
            if len(_semantic_cache) % 50 == 0:
                self.save_cache()
        
        return score
    
    def calculate_difflib_similarity(self, point: str, criteria_name: str) -> float:
        """Smart difflib-based similarity calculation optimized for Arabic text"""
        try:
            # Clean and normalize text for better comparison
            point_clean = re.sub(r'[^\w\s]', ' ', point.lower()).strip()
            criteria_clean = re.sub(r'[^\w\s]', ' ', criteria_name.lower()).strip()
            
            if not point_clean or not criteria_clean:
                return 0.0
            
            # Method 1: Sequence matcher for overall similarity
            seq_matcher = difflib.SequenceMatcher(None, point_clean, criteria_clean)
            sequence_similarity = seq_matcher.ratio()
            
            # Method 2: Word-level comparison using get_close_matches
            point_words = point_clean.split()
            criteria_words = criteria_clean.split()
            
            word_matches = 0
            total_words = len(point_words)
            
            if total_words > 0:
                for word in point_words:
                    if len(word) > 2:  # Skip very short words
                        # Find close matches with generous cutoff for Arabic variations
                        close_matches = difflib.get_close_matches(
                            word, criteria_words, 
                            n=1,           # Get best match only
                            cutoff=0.6     # 60% similarity threshold
                        )
                        if close_matches:
                            word_matches += 1
                
                word_similarity = word_matches / total_words
            else:
                word_similarity = 0.0
            
            # Method 3: Quick ratio for performance (faster than ratio())
            quick_similarity = seq_matcher.quick_ratio()
            
            # Combine the three methods with weights
            final_score = (sequence_similarity * 0.4) + (word_similarity * 0.4) + (quick_similarity * 0.2)
            
            return min(final_score, 1.0)
            
        except Exception:
            # Fallback to simple comparison if difflib fails
            if point.lower() in criteria_name.lower() or criteria_name.lower() in point.lower():
                return 0.3
            return 0.0

# Global semantic matcher instance
semantic_matcher = SimpleSemanticMatcher()


def get_criteria_for_client(client_id: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieve the criteria list for the given client from MongoDB with caching.

    Uses LRU cache to avoid repeated database queries for the same client.
    Cache expires after 15 minutes to ensure data freshness.
    """
    current_time = time.time()
    
    # Check cache first
    with _cache_lock:
        if client_id in _criteria_cache:
            cached_data, timestamp = _criteria_cache[client_id]
            if current_time - timestamp < _cache_timeout:
                return cached_data
            else:
                # Remove expired cache entry
                del _criteria_cache[client_id]
    
    from bson import ObjectId
    try:
        # Try to fetch by ObjectId first; fallback to string
        criteria_doc = None
        try:
            criteria_doc = criteria_collection.find_one(
                {"_id": ObjectId(client_id)},
                {"criteria": 1}  # Only fetch criteria field
            )
        except Exception:
            criteria_doc = criteria_collection.find_one(
                {"_id": client_id},
                {"criteria": 1}
            )
        
        result = None
        if criteria_doc and "criteria" in criteria_doc:
            result = criteria_doc["criteria"]
        
        # Cache the result
        with _cache_lock:
            _criteria_cache[client_id] = (result, current_time)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get criteria for client {client_id}: {e}")
        return None


# LRU cache for classification results (increased size for speed)
@lru_cache(maxsize=2000)  # Increased cache size significantly
def _cached_classify_point(point_hash: str, criteria_names_str: str) -> Optional[str]:
    """Cached classification to avoid repeated API calls for same points."""
    criteria_names = criteria_names_str.split('|||')
    
    # Use asyncio.run for sync compatibility in cached function
    try:
        return asyncio.run(_async_classify_point_cached(point_hash, criteria_names))
    except Exception:
        return None

async def _async_classify_point_cached(point_text: str, criteria_names: List[str]) -> Optional[str]:
    """Async LLM classification with optimized parameters"""
    llama_url = "http://localhost:11434/api/generate"
    llama_model = "finalend/llama-3.1-storm:8b"
    
    # Ultra-short prompt for maximum speed
    prompt = f"النقطة: {point_text[:40]}\nالمعايير: {', '.join(criteria_names[:4])}\nأفضل معيار:"
    
    data = {
        "model": llama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,   # Lower for speed and consistency
            "num_predict": 10,    # Minimal tokens
            "num_ctx": 256,       # Much smaller context
            "top_k": 2,           # Very focused
            "top_p": 0.7
        }
    }
    
    try:
        session = await get_async_session()
        async with session.post(llama_url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("response", "").strip()
            else:
                return None
    except asyncio.TimeoutError:
        logger.info("Ollama timeout - using enhanced fallbacks")
        return None
    except Exception as e:
        logger.info(f"Ollama error: {str(e)} - using enhanced fallbacks")
        return None


def find_closest_criteria_rapidfuzz(text: str, criteria_names: List[str], threshold: float = 0.6) -> Optional[str]:
    """Ultra-fast criteria matching using rapidfuzz with priority keywords"""
    try:
        if not text or not criteria_names:
            return None
        
        clean_text = text.lower().strip()
        if not clean_text:
            return None
        
        # FAST: Priority keyword matching first
        text_keywords = set(clean_text.split())
        for text_word in text_keywords:
            for main_keyword, variations in PRIORITY_KEYWORDS.items():
                if text_word in variations:
                    # Look for criteria containing the main keyword using rapidfuzz
                    keyword_matches = rapidfuzz.process.extract(
                        main_keyword, criteria_names, limit=1, score_cutoff=50
                    )
                    if keyword_matches:
                        return keyword_matches[0][0]
        
        # MEDIUM: Use rapidfuzz for general matching
        matches = rapidfuzz.process.extract(
            clean_text, criteria_names, limit=1, score_cutoff=threshold * 100
        )
        
        if matches:
            return matches[0][0]
        
        # SLOW: Fallback to partial matching
        best_match = None
        best_score = 0.0
        
        for criteria in criteria_names:
            # Use rapidfuzz token ratio for better Arabic support
            score = rapidfuzz.fuzz.token_ratio(clean_text, criteria.lower()) / 100.0
            if score > best_score and score >= threshold:
                best_score = score
                best_match = criteria
        
        return best_match
        
    except Exception as e:
        logger.warning(f"rapidfuzz matching failed: {e}")
        return None


def classify_point_enhanced(point: str, criteria_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Enhanced 8-tier classification using multiple methods with difflib intelligence"""
    try:
        if not point or not criteria_list:
            return None
            
        criteria_names = [c['name'] for c in criteria_list if 'name' in c]
        if not criteria_names:
            return None
        
        # Method 0: SMART rapidfuzz quick match for exact/near-exact matches (NEW!)
        rapidfuzz_match = find_closest_criteria_rapidfuzz(point, criteria_names, threshold=0.7)  # High threshold for early exit
        if rapidfuzz_match:
            # Find the criteria object that matches
            for c in criteria_list:
                if c.get("name", "").lower() == rapidfuzz_match.lower():
                    return c
        
        # Method 1: Check LRU cache first (fastest)
        point_hash = str(hash(point.lower().strip()))
        criteria_names_str = '|||'.join(criteria_names)
        
        try:
            cached_result = _cached_classify_point(point_hash, criteria_names_str)
            if cached_result:
                for c in criteria_list:
                    if cached_result.lower() in c.get("name", "").lower():
                        return c
        except Exception:
            # Cache failed, continue to other methods
            pass
        
        # Method 2: ENHANCED Quick string matching with delivery service priority
        point_lower = point.lower().strip()
        
        # PRIORITY 1: Check for delivery service first if point mentions delivery
        delivery_keywords = ["توصيل", "وصل", "وصول", "تسليم", "موعد", "الوقت المحدد", "بالموعد", "في الوقت"]
        if any(keyword in point_lower for keyword in delivery_keywords):
            for c in criteria_list:
                criteria_name_lower = c.get("name", "").lower()
                if "توصيل" in criteria_name_lower or "تسليم" in criteria_name_lower or "delivery" in criteria_name_lower:
                    return c
        
        # PRIORITY 2: Check for taste/food-related criteria
        taste_keywords = ["طعم", "طعام", "مذاق", "نكهة", "لذيذ", "طري", "مالح", "حلو", "مر"]
        if any(keyword in point_lower for keyword in taste_keywords):
            for c in criteria_list:
                criteria_name_lower = c.get("name", "").lower()
                if "طعم" in criteria_name_lower or "طعام" in criteria_name_lower or "taste" in criteria_name_lower or "food" in criteria_name_lower:
                    return c
        
        # PRIORITY 3: Check for price-related criteria
        price_keywords = ["سعر", "ثمن", "تكلفة", "غالي", "رخيص", "مرتفع", "منخفض", "price", "cost"]
        if any(keyword in point_lower for keyword in price_keywords):
            for c in criteria_list:
                criteria_name_lower = c.get("name", "").lower()
                if "سعر" in criteria_name_lower or "ثمن" in criteria_name_lower or "price" in criteria_name_lower or "cost" in criteria_name_lower:
                    return c
        
        # PRIORITY 4: Check for customer service criteria
        service_keywords = ["خدمة العملاء", "تعامل", "فريق", "موظف", "support", "customer service"]
        if any(keyword in point_lower for keyword in service_keywords):
            for c in criteria_list:
                criteria_name_lower = c.get("name", "").lower()
                if "خدمة العملاء" in criteria_name_lower or "customer" in criteria_name_lower or "support" in criteria_name_lower:
                    return c
        
        # Regular string matching for other cases
        for c in criteria_list:
            criteria_name_lower = c.get("name", "").lower()
            if (criteria_name_lower in point_lower or 
                point_lower in criteria_name_lower or
                any(word in criteria_name_lower for word in point_lower.split()[:3])):
                return c
        
        # Method 3: Enhanced semantic matching (fast, no external libs) - IMPROVED THRESHOLDS
        best_match = None
        best_score = 0.0
        
        for c in criteria_list:
            criteria_name = c.get("name", "")
            similarity_score = semantic_matcher.calculate_similarity(point, criteria_name)
            
            # Dynamic threshold based on point length and content
            threshold = 0.3 if len(point_lower) > 20 else 0.25  # Lower threshold for shorter points
            
            if similarity_score > best_score and similarity_score > threshold:
                best_score = similarity_score
                best_match = c
        
        if best_match and best_score > 0.35:  # Only return if confidence is good
            return best_match
        
        # Method 4: Enhanced Word overlap analysis with Arabic-specific matching
        for c in criteria_list:
            criteria_name_lower = c.get("name", "").lower()
            point_words = set(point_lower.split()[:5])
            criteria_words = set(criteria_name_lower.split())
            
            # Enhanced scoring with partial matches
            overlap = len(point_words.intersection(criteria_words))
            total_words = len(point_words.union(criteria_words))
            
            # Check for partial word matches (important for Arabic)
            partial_matches = 0
            for p_word in point_words:
                for c_word in criteria_words:
                    if len(p_word) > 3 and len(c_word) > 3:
                        if p_word in c_word or c_word in p_word:
                            partial_matches += 0.5
            
            if total_words > 0:
                base_score = overlap / total_words
                enhanced_score = base_score + (partial_matches / max(len(criteria_words), 1))
                
                if enhanced_score > best_score and enhanced_score > 0.15:  # Lowered threshold
                    best_score = enhanced_score
                    best_match = c
        
        if best_match and best_score > 0.2:
            return best_match
        
        # Method 4.5: ENHANCED rapidfuzz matching with lower threshold for tough cases
        if best_score < 0.25:  # Only when other methods are struggling
            rapidfuzz_match_lower = find_closest_criteria_rapidfuzz(point, criteria_names, threshold=0.3)  # Lower threshold
            if rapidfuzz_match_lower:
                for c in criteria_list:
                    if c.get("name", "").lower() == rapidfuzz_match_lower.lower():
                        return c
        
        # Method 6: Advanced keyword-based fallback with Arabic context understanding
        point_keywords = point_lower.split()
        best_keyword_match = None
        best_keyword_score = 0.0
        
        for c in criteria_list:
            criteria_keywords = c.get("name", "").lower().split()
            score = 0.0
            
            # Exact word matches
            for p_word in point_keywords:
                for c_word in criteria_keywords:
                    if len(p_word) > 3 and len(c_word) > 3:
                        if p_word == c_word:
                            score += 1.0  # Exact match bonus
                        elif p_word in c_word or c_word in p_word:
                            score += 0.7  # Partial match
                        elif abs(len(p_word) - len(c_word)) <= 2:
                            # Check for similar length words (Arabic variations)
                            common_chars = set(p_word) & set(c_word)
                            if len(common_chars) >= min(len(p_word), len(c_word)) * 0.6:
                                score += 0.4
            
            # Normalize score by criteria length
            normalized_score = score / max(len(criteria_keywords), 1)
            
            if normalized_score > best_keyword_score:
                best_keyword_score = normalized_score
                best_keyword_match = c
        
        if best_keyword_match and best_keyword_score > 0.3:
            return best_keyword_match
        
        # Method 7: Enhanced LLM usage with classify_point_with_llama for challenging cases
        if best_score < 0.25 and len(point_lower) > 10:  # Only for substantial points
            try:
                # Use classify_point_with_llama for better accuracy
                llm_result = classify_point_with_llama(point, criteria_list)
                
                if llm_result and "criteria_id" in llm_result:
                    # Find the criteria by ID
                    for c in criteria_list:
                        if c.get("id") == llm_result["criteria_id"]:
                            return c
                            
                # Fallback to name matching if ID not found
                if llm_result and "criteria_name" in llm_result:
                    criteria_name = llm_result["criteria_name"]
                    for c in criteria_list:
                        if criteria_name.lower() in c.get("name", "").lower():
                            return c
                            
            except Exception as e:
                # LLM call failed, try cached version
                try:
                    llm_result = _cached_classify_point(point_hash, criteria_names_str)
                    if llm_result:
                        for c in criteria_list:
                            if llm_result.lower() in c.get("name", "").lower():
                                return c
                except Exception:
                    # Both LLM and cache failed, continue to final fallback
                    pass
        
        # Method 8: Final fallback - return best match if any reasonable confidence
        if best_match and best_score > 0.1:
            return best_match
        elif best_keyword_match and best_keyword_score > 0.2:
            return best_keyword_match
        
        return None
        
    except Exception:
        return None



def link_points_to_criteria(points: List[Any], criteria_list: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Attach criteria information to each extracted feedback point with parallel processing.

    Enhanced version that processes multiple points concurrently for better performance.
    Uses ThreadPoolExecutor to parallelize the classification API calls.
    """
    if not points or not criteria_list:
        return []
    
    # Normalize points first (sequential, fast operation)
    normalized_points = []
    for item in points:
        if isinstance(item, str):
            text = item.strip()[:200]  # Limit text length for speed
            original_item = item
        elif isinstance(item, dict):
            text = ""
            for key in ["text", "point", "content", "message"]:
                if key in item and isinstance(item[key], str):
                    text = item[key].strip()[:200]  # Limit text length for speed
                    break
            if not text:
                text = str(item)[:200]
            original_item = item
        else:
            text = str(item)[:200]
            original_item = item
        
        if text and len(text) > 3:  # Only process meaningful texts
            normalized_points.append((text, original_item))
    
    if not normalized_points:
        return []
    
    linked: List[Dict[str, Any]] = []
    
    # For small number of points, process sequentially to avoid overhead
    if len(normalized_points) <= 2:  # Reduced threshold for faster processing
        for text, original_item in normalized_points:
            matched = classify_point_enhanced(text, criteria_list)  # Use enhanced classifier
            
            # CRITICAL: Apply sentiment override during point processing with double-check
            sentiment_override = semantic_matcher.detect_sentiment_override(text)
            
            # Double-check the sentiment based on the context (positive vs negative points)
            # If this point came from positive_points but override says negative, trust the override
            # If this point came from negative_points but override says positive, trust the override
            
            linked_point = {
                "point": text,
                "criteria_id": matched.get("id") if matched else None,
                "criteria_name": matched.get("name") if matched else None,
                "criteria_weight": matched.get("weight", 0.0) if matched else 0.0,
                "sentiment_override": sentiment_override  # Add sentiment flag
            }
            # Copy other keys if input was a dict (optimized)
            if isinstance(original_item, dict):
                for k, v in original_item.items():
                    if k not in {"text", "point", "content", "message"}:  # Use set for faster lookup
                        linked_point[k] = v
            linked.append(linked_point)
    else:
        # Parallel processing for larger lists with increased workers
        def process_point(text_and_item):
            text, original_item = text_and_item
            try:
                matched = classify_point_enhanced(text, criteria_list)  # Use enhanced classifier
                
                # CRITICAL: Apply sentiment override during point processing
                sentiment_override = semantic_matcher.detect_sentiment_override(text)
                
                linked_point = {
                    "point": text,
                    "criteria_id": matched.get("id") if matched else None,
                    "criteria_name": matched.get("name") if matched else None,
                    "criteria_weight": matched.get("weight", 0.0) if matched else 0.0,
                    "sentiment_override": sentiment_override  # Add sentiment flag for later correction
                }
                # Copy other keys if input was a dict (optimized)
                if isinstance(original_item, dict):
                    for k, v in original_item.items():
                        if k not in {"text", "point", "content", "message"}:
                            linked_point[k] = v
                return linked_point
            except Exception:
                # Silent failure for individual points
                return {
                    "point": text,
                    "criteria_id": None,
                    "criteria_name": None,
                    "criteria_weight": 0.0,
                    "sentiment_override": "neutral"
                }
        
        # Use ThreadPoolExecutor for parallel processing with higher concurrency
        max_workers = min(max_workers * 2, len(normalized_points), 8)  # Increased max workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_point = {
                executor.submit(process_point, point_data): point_data 
                for point_data in normalized_points
            }
            
            for future in as_completed(future_to_point):
                try:
                    result = future.result(timeout=25)  # Reduced timeout for speed
                    linked.append(result)
                except Exception:
                    # Silent failure with fallback
                    point_data = future_to_point[future]
                    linked.append({
                        "point": point_data[0],
                        "criteria_id": None,
                        "criteria_name": None,
                        "criteria_weight": 0.0,
                        "sentiment_override": "neutral"
                    })
    
    return linked


def apply_sentiment_overrides(positive_points: List[Dict[str, Any]], negative_points: List[Dict[str, Any]]) -> tuple:
    """Apply sentiment overrides to move misclassified points early in the process"""
    corrected_positive = []
    corrected_negative = []
    
    # Check positive points for negative sentiment overrides
    for point in positive_points:
        sentiment = point.get("sentiment_override", "neutral")
        if sentiment == "negative":
            logger.info(f"EARLY CORRECTION: Moving '{point.get('point', '')}' from positive to negative")
            corrected_negative.append(point)
        else:
            corrected_positive.append(point)
    
    # Check negative points for positive sentiment overrides  
    for point in negative_points:
        sentiment = point.get("sentiment_override", "neutral")
        if sentiment == "positive":
            logger.info(f"EARLY CORRECTION: Moving '{point.get('point', '')}' from negative to positive")
            corrected_positive.append(point)
        else:
            corrected_negative.append(point)
    
    return corrected_positive, corrected_negative


def correct_sentiment_classification(positive_points: List[Dict[str, Any]], negative_points: List[Dict[str, Any]]) -> tuple:
    """Critical function to fix sentiment misclassification using enhanced detection.
    
    This addresses the core issue where positive feedback like "خدمة التوصيل كانت على الوقت المحدد" 
    gets incorrectly classified as negative, and negative feedback gets misclassified as positive.
    """
    corrected_positive = []
    corrected_negative = []
    
    logger.info("🔍 Starting enhanced sentiment correction...")
    
    # Process positive points - check for misclassified negatives
    for item in positive_points:
        points_to_move = []
        remaining_points = []
        
        for point in item.get("points", []):
            # Enhanced sentiment detection
            sentiment = semantic_matcher.detect_sentiment_override(point)
            
            # If sentiment is clearly negative, move it
            if sentiment == "negative":
                points_to_move.append(point)
                logger.info(f"✅ CORRECTION: Moving clearly negative point from positive list: '{point}'")
            else:
                remaining_points.append(point)
        
        # If we have remaining positive points, keep the item
        if remaining_points:
            corrected_item = item.copy()
            corrected_item["points"] = remaining_points
            corrected_item["count"] = len(remaining_points)
            corrected_positive.append(corrected_item)
        
        # Add moved points to negative with same criteria
        if points_to_move:
            moved_item = {
                "criteria_id": item.get("criteria_id"),
                "criteria_name": item.get("criteria_name"), 
                "criteria_weight": item.get("criteria_weight", 0.0),
                "points": points_to_move,
                "count": len(points_to_move)
            }
            corrected_negative.append(moved_item)
    
    # Process negative points - check for misclassified positives  
    for item in negative_points:
        points_to_move = []
        remaining_points = []
        
        for point in item.get("points", []):
            # Enhanced sentiment detection
            sentiment = semantic_matcher.detect_sentiment_override(point)
            
            # If sentiment is clearly positive, move it
            if sentiment == "positive":
                points_to_move.append(point)
                logger.info(f"✅ CORRECTION: Moving clearly positive point from negative list: '{point}'")
            else:
                remaining_points.append(point)
        
        # If we have remaining negative points, keep the item
        if remaining_points:
            corrected_item = item.copy()
            corrected_item["points"] = remaining_points
            corrected_item["count"] = len(remaining_points)
            corrected_negative.append(corrected_item)
        
        # Add moved points to positive with same criteria
        if points_to_move:
            moved_item = {
                "criteria_id": item.get("criteria_id"),
                "criteria_name": item.get("criteria_name"),
                "criteria_weight": item.get("criteria_weight", 0.0), 
                "points": points_to_move,
                "count": len(points_to_move)
            }
            corrected_positive.append(moved_item)
    
    # Merge items with same criteria_id in each list
    def merge_same_criteria(items_list):
        merged = {}
        for item in items_list:
            key = (item.get("criteria_id"), item.get("criteria_name"))
            if key in merged:
                merged[key]["points"].extend(item.get("points", []))
                merged[key]["count"] += item.get("count", 0)
                merged[key]["criteria_weight"] += item.get("criteria_weight", 0.0)
            else:
                merged[key] = item.copy()
        return list(merged.values())
    
    final_positive = merge_same_criteria(corrected_positive)
    final_negative = merge_same_criteria(corrected_negative)
    
    return final_positive, final_negative


def collapse_points_by_criteria(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate feedback points by criteria with optimized performance.

    Enhanced version with better memory usage and faster aggregation logic.
    """
    if not points:
        return []
    
    aggregated: Dict[Any, Dict[str, Any]] = {}
    
    for p in points:
        cid = p.get("criteria_id")
        cname = p.get("criteria_name")
        weight = p.get("criteria_weight", 0.0)
        text = p.get("point", "")
        
        if cid is None:
            # Points with no criteria are kept separate
            key = (None, text)
            if key not in aggregated:
                aggregated[key] = {
                    "criteria_id": None,
                    "criteria_name": None,
                    "criteria_weight": 0.0,
                    "points": [text],
                    "count": 1
                }
            else:
                aggregated[key]["points"].append(text)
                aggregated[key]["count"] += 1
        else:
            key = (cid, cname)
            if key not in aggregated:
                aggregated[key] = {
                    "criteria_id": cid,
                    "criteria_name": cname,
                    "criteria_weight": weight,
                    "points": [text],
                    "count": 1
                }
            else:
                aggregated[key]["criteria_weight"] += weight
                aggregated[key]["points"].append(text)
                aggregated[key]["count"] += 1
    
    # Convert to list more efficiently
    return list(aggregated.values())


def save_analysis_result(uuid: str, analysis_object: Dict[str, Any]) -> bool:
    """Save the analysis results in MongoDB.

    Returns True if the document is updated or inserted successfully.
    """
    try:
        if not uuid or not isinstance(uuid, str):
            return False
        if not analysis_object or not isinstance(analysis_object, dict):
            return False
        result = results_collection.update_one(
            {"uuid": uuid},
            {"$set": {
                "analysis": analysis_object,
                "updated_at": datetime.now()
            }},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None
    except Exception as e:
        logger.error(f"Failed to save analysis for UUID {uuid}: {e}")
        return False


async def analyze_and_calculate_scores_async(transcript: str, client_id: str, max_workers: int = 8) -> Dict[str, Any]:
    """Ultra-fast async analysis with concurrent operations and enhanced caching"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting ultra-fast async analysis for client: {client_id}")

        # 1. Fetch criteria asynchronously
        criteria_list = get_criteria_for_client(client_id)
        if not criteria_list:
            return {
                "error": "No criteria found for this client.",
                "processing_time": time.time() - start_time
            }

        # 2. Extract feedback points using feedback_points_extractor
        # Limit transcript length for faster processing
        transcript_text = transcript[:1500] if isinstance(transcript, str) else str(transcript)[:1500]
        
        # Use feedback points extractor with LLaMA
        logger.info("🚀 Using feedback points extractor with LLaMA...")
        extraction_start = time.time()
        
        try:
            points_data = extract_points_with_llama(transcript_text)
            points_result = PointsAnalysisResult(
                analysis=points_data,
                processing_time=time.time() - extraction_start,
                confidence=0.9
            )
            
            # ✅ معالجة أذكى للنقاط المستخرجة - تحقق من جودة الفصل
            logger.info(f"🔍 Raw extraction result: pos={len(points_data.get('positive_points', []))}, neg={len(points_data.get('negative_points', []))}")
            
        except Exception as e:
            logger.error(f"❌ Feedback points extraction failed: {e}")
            # استخدام fallback محلي أكثر ذكاءً
            points_result = PointsAnalysisResult(
                analysis={
                    "positive_points": [],
                    "negative_points": [f"خطأ في التحليل: {str(e)[:50]}"]
                },
                error=str(e),
                processing_time=time.time() - extraction_start
            )
        
        if points_result.error:
            logger.warning(f"⚠️ Points extraction had issues: {points_result.error}")
        
        # ✅ فحص ذكي للنقاط المستخرجة وتحسينها إذا لزم الأمر
        raw_positive = points_result.analysis.get("positive_points", [])[:10]
        raw_negative = points_result.analysis.get("negative_points", [])[:10]
        
        # إذا كانت النقاط طويلة جداً أو غير مفصلة بشكل صحيح، قم بالتحسين
        def enhance_point_separation(positive_points, negative_points, original_text):
            """تحسين فصل النقاط إذا كانت غير مفصلة بشكل صحيح"""
            enhanced_positive = []
            enhanced_negative = []
            
            # فحص إذا كانت النقاط طويلة جداً (تشير لعدم الفصل الصحيح)
            for point in positive_points:
                if isinstance(point, str) and len(point) > 100:
                    # النقطة طويلة جداً، قم بتقسيمها
                    logger.warning(f"⚠️ Long positive point detected, enhancing: {point[:50]}...")
                    # استخرج الكلمات الإيجابية الأساسية
                    positive_keywords = ["ممتاز", "رائع", "زاكي", "رايق", "حلو", "جيد", "مريح", "نظيف", 
                                       "عالم آخر", "تفتح النفس", "محافظين على الجودة", "خياري الأول"]
                    
                    found_positives = []
                    for keyword in positive_keywords:
                        if keyword in point.lower():
                            # استخرج الجملة المحيطة بالكلمة الإيجابية
                            sentences = point.split('.')
                            for sentence in sentences:
                                if keyword in sentence.lower() and len(sentence.strip()) > 5:
                                    clean_sentence = sentence.strip()
                                    if len(clean_sentence) > 10:
                                        found_positives.append(clean_sentence[:60])
                    
                    if found_positives:
                        enhanced_positive.extend(found_positives[:3])  # حد أقصى 3 نقاط
                    else:
                        # احتفظ بالنقطة الأصلية مختصرة
                        enhanced_positive.append(point[:60] + "...")
                else:
                    enhanced_positive.append(point)
            
            # نفس المعالجة للنقاط السلبية
            for point in negative_points:
                if isinstance(point, str) and len(point) > 100:
                    logger.warning(f"⚠️ Long negative point detected, enhancing: {point[:50]}...")
                    negative_keywords = ["سيء", "بطيء", "متأخر", "مش كويس", "غالي", "مشكلة", 
                                       "أبدا ما عجبني", "لم يعجبني", "محبط", "مزعج"]
                    
                    found_negatives = []
                    for keyword in negative_keywords:
                        if keyword in point.lower():
                            sentences = point.split('.')
                            for sentence in sentences:
                                if keyword in sentence.lower() and len(sentence.strip()) > 5:
                                    clean_sentence = sentence.strip()
                                    if len(clean_sentence) > 10:
                                        found_negatives.append(clean_sentence[:60])
                    
                    if found_negatives:
                        enhanced_negative.extend(found_negatives[:3])
                    else:
                        enhanced_negative.append(point[:60] + "...")
                else:
                    enhanced_negative.append(point)
            
            return enhanced_positive, enhanced_negative
        
        # تطبيق التحسين إذا لزم الأمر
        enhanced_positive, enhanced_negative = enhance_point_separation(raw_positive, raw_negative, transcript_text)
        
        # استبدال النقاط بالنسخة المحسنة
        raw_positive = enhanced_positive
        raw_negative = enhanced_negative
        
        logger.info(f"⚡ Async points extracted in {points_result.processing_time:.3f}s: {len(raw_positive)} pos, {len(raw_negative)} neg")

        # 3. Process points with increased concurrency
        extraction_start = time.time()
        
        # Always use parallel processing for better performance
        with ThreadPoolExecutor(max_workers=min(max_workers + 4, 12)) as executor:
            # Submit both tasks with higher worker counts
            pos_future = executor.submit(link_points_to_criteria, raw_positive, criteria_list, max_workers + 4)
            neg_future = executor.submit(link_points_to_criteria, raw_negative, criteria_list, max_workers + 4)
            
            # Get results with extended timeout
            positive_points = pos_future.result(timeout=45)
            negative_points = neg_future.result(timeout=45)
        
        # Apply early sentiment correction
        positive_points, negative_points = apply_sentiment_overrides(positive_points, negative_points)
        
        extraction_time = time.time() - extraction_start

        # 4. Fast aggregation and correction
        collapse_start = time.time()
        pos_collapsed = collapse_points_by_criteria(positive_points)
        neg_collapsed = collapse_points_by_criteria(negative_points)
        
        # Sentiment correction - CRITICAL for proper classification
        correction_start = time.time()
        pos_collapsed, neg_collapsed = correct_sentiment_classification(pos_collapsed, neg_collapsed)
        correction_time = time.time() - correction_start
        collapse_time = time.time() - collapse_start

        # 5. Calculate scores using enhanced algorithm
        calc_start = time.time()
        
        # Calculate total weights properly
        total_positive_weight = sum(
            item.get("criteria_weight", 0.0) 
            for item in pos_collapsed 
            if item.get("criteria_id") is not None
        )
        
        total_negative_weight = sum(
            item.get("criteria_weight", 0.0) 
            for item in neg_collapsed 
            if item.get("criteria_id") is not None
        )
        
        # Enhanced scoring system with proper percentage calculation
        total_weight = total_positive_weight + total_negative_weight
        
        if total_weight > 0:
            # Calculate percentages based on weight distribution
            positive_percentage = (total_positive_weight / total_weight) * 100
            negative_percentage = (total_negative_weight / total_weight) * 100
        else:
            # If no weighted points, use point counts
            pos_count = sum(item.get("count", 0) for item in pos_collapsed)
            neg_count = sum(item.get("count", 0) for item in neg_collapsed)
            total_count = pos_count + neg_count
            
            if total_count > 0:
                positive_percentage = (pos_count / total_count) * 100
                negative_percentage = (neg_count / total_count) * 100
            else:
                positive_percentage = 50.0
                negative_percentage = 50.0
        
        
        total_gray_weight = max(1.0 - (total_positive_weight + total_negative_weight), 0.0)
        
        # Sentiment classification based on percentages
        if total_weight == 0:
            sentiment_label = "neutral"
        elif negative_percentage >= 60:
            sentiment_label = "negative"
        elif positive_percentage >= 60:
            sentiment_label = "positive"
        else:
            sentiment_label = "mixed"
            
        calc_time = time.time() - calc_start

        # 6. Build optimized analysis object with proper structure
        total_processing_time = time.time() - start_time
        
        pos_count = sum(item.get("count", 0) for item in pos_collapsed)
        neg_count = sum(item.get("count", 0) for item in neg_collapsed)

        analysis_object = {
            "positive_points": pos_collapsed,
            "negative_points": neg_collapsed,
            "classification": sentiment_label,
            "scores": {
                "total_positive_weight": round(total_positive_weight, 3),
                "total_negative_weight": round(total_negative_weight, 3),
                "total_gray_weight": round(total_gray_weight, 3),
                "positive_percentage": f"{round(positive_percentage, 1)}%",
                "negative_percentage": f"{round(negative_percentage, 1)}%",
                "total_weight": round(total_weight, 3)
            },
            "metadata": {
                "total_positive_points": pos_count,
                "total_negative_points": neg_count,
                "total_points": pos_count + neg_count,
                "analysis_timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "extractor_type": "ultra_fast_async_with_feedback_points_extractor",
                "system_status": {
                    "async_processing": True,
                    "feedback_extractor_enabled": True,
                    "motor_mongodb": True,
                    "extraction_successful": points_result.error is None,
                    "classification_method": "async_hybrid_with_sentiment_correction"
                },
                "performance": {
                    "total_processing_time": round(total_processing_time, 3),
                    "extraction_time": round(extraction_time, 3),
                    "collapse_time": round(collapse_time, 3),
                    "calculation_time": round(calc_time, 3),
                    "correction_time": round(correction_time, 3),
                    "points_extraction_time": round(points_result.processing_time, 3),
                    "concurrent_processing": True,
                    "max_workers_used": max_workers + 4
                }
            }
        }        # 7. Save asynchronously without blocking
        save_analysis_result(str(client_id), analysis_object)

        return {
            "success": True,
            "analysis": analysis_object,
            "positive_score": f"{round(positive_percentage, 1)}%",
            "negative_score": f"{round(negative_percentage, 1)}%",
            "processing_time": round(total_processing_time, 2)
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Async analysis failed: {e}")
        return {
            "error": "Async analysis processing failed",
            "processing_time": round(error_time, 2)
        }

def analyze_and_calculate_scores(transcript: str, client_id: str, max_workers: int = 6) -> Dict[str, Any]:
    """Analyze a transcript and compute sentiment scores with enhanced performance.

    Enhanced version with parallel processing, caching, and optimized database operations.
    Uses context manager for proper resource management and includes performance metrics.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting enhanced analysis for client: {client_id}")

        # 1. Fetch criteria for client (with caching)
        criteria_list = get_criteria_for_client(client_id)
        if not criteria_list:
            return {
                "error": "No criteria found for this client.",
                "processing_time": time.time() - start_time
            }

        # 2. Extract feedback points using feedback_points_extractor
        # Ensure input is string and limit length for speed
        transcript_text = transcript[:1500] if isinstance(transcript, str) else str(transcript)[:1500]
        
        # Use feedback points extractor with LLaMA
        logger.info("🚀 Using feedback points extractor with LLaMA...")
        extraction_start = time.time()
        
        try:
            points_data = extract_points_with_llama(transcript_text)
            points_result = PointsAnalysisResult(
                analysis=points_data,
                processing_time=time.time() - extraction_start,
                confidence=0.9
            )
            
            # ✅ معالجة أذكى للنقاط المستخرجة - تحقق من جودة الفصل
            logger.info(f"🔍 Raw extraction result: pos={len(points_data.get('positive_points', []))}, neg={len(points_data.get('negative_points', []))}")
            
        except Exception as e:
            logger.error(f"❌ Feedback points extraction failed: {e}")
            # استخدام fallback محلي أكثر ذكاءً
            points_result = PointsAnalysisResult(
                analysis={
                    "positive_points": [],
                    "negative_points": [f"خطأ في التحليل: {str(e)[:50]}"]
                },
                error=str(e),
                processing_time=time.time() - extraction_start
            )
        
        # Convert to expected format
        if points_result.error:
            logger.warning(f"⚠️ Points extraction had issues: {points_result.error}")
        
        # ✅ فحص ذكي للنقاط المستخرجة وتحسينها إذا لزم الأمر
        raw_positive = points_result.analysis.get("positive_points", [])[:10]
        raw_negative = points_result.analysis.get("negative_points", [])[:10]
        
        # إذا كانت النقاط طويلة جداً أو غير مفصلة بشكل صحيح، قم بالتحسين
        def enhance_point_separation_sync(positive_points, negative_points, original_text):
            """تحسين فصل النقاط إذا كانت غير مفصلة بشكل صحيح"""
            enhanced_positive = []
            enhanced_negative = []
            
            # فحص إذا كانت النقاط طويلة جداً (تشير لعدم الفصل الصحيح)
            for point in positive_points:
                if isinstance(point, str) and len(point) > 100:
                    # النقطة طويلة جداً، قم بتقسيمها
                    logger.warning(f"⚠️ Long positive point detected, enhancing: {point[:50]}...")
                    # استخرج الكلمات الإيجابية الأساسية
                    positive_keywords = ["ممتاز", "رائع", "زاكي", "رايق", "حلو", "جيد", "مريح", "نظيف", 
                                       "عالم آخر", "تفتح النفس", "محافظين على الجودة", "خياري الأول",
                                       "غيره", "ليس أي", "مميز"]
                    
                    found_positives = []
                    # تقسيم النقطة لجمل
                    sentences = point.replace('،', '.').split('.')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10:
                            for keyword in positive_keywords:
                                if keyword in sentence.lower():
                                    found_positives.append(sentence[:60])
                                    break
                    
                    if found_positives:
                        enhanced_positive.extend(found_positives[:3])  # حد أقصى 3 نقاط
                    else:
                        # احتفظ بالنقطة الأصلية مختصرة
                        enhanced_positive.append(point[:60] + "...")
                else:
                    enhanced_positive.append(point)
            
            # نفس المعالجة للنقاط السلبية
            for point in negative_points:
                if isinstance(point, str) and len(point) > 100:
                    logger.warning(f"⚠️ Long negative point detected, enhancing: {point[:50]}...")
                    negative_keywords = ["سيء", "بطيء", "متأخر", "مش كويس", "غالي", "مشكلة", 
                                       "أبدا ما عجبني", "لم يعجبني", "محبط", "مزعج"]
                    
                    found_negatives = []
                    sentences = point.replace('،', '.').split('.')
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10:
                            for keyword in negative_keywords:
                                if keyword in sentence.lower():
                                    found_negatives.append(sentence[:60])
                                    break
                    
                    if found_negatives:
                        enhanced_negative.extend(found_negatives[:3])
                    else:
                        enhanced_negative.append(point[:60] + "...")
                else:
                    enhanced_negative.append(point)
            
            return enhanced_positive, enhanced_negative
        
        # تطبيق التحسين إذا لزم الأمر
        enhanced_positive, enhanced_negative = enhance_point_separation_sync(raw_positive, raw_negative, transcript_text)
        
        # استبدال النقاط بالنسخة المحسنة
        raw_positive = enhanced_positive
        raw_negative = enhanced_negative
        
        logger.info(f"⚡ Points extracted in {points_result.processing_time:.3f}s: {len(raw_positive)} pos, {len(raw_negative)} neg")

        # 3. Link points to criteria with parallel processing AND apply smart sentiment separation
        extraction_start = time.time()
        
        # Process positive and negative points in parallel if there are enough points
        if len(raw_positive) + len(raw_negative) > 4:  # Lower threshold for faster parallel start
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks with increased workers
                pos_future = executor.submit(link_points_to_criteria, raw_positive, criteria_list, max_workers + 2)
                neg_future = executor.submit(link_points_to_criteria, raw_negative, criteria_list, max_workers + 2)
                
                # Get results with shorter timeout
                positive_points = pos_future.result(timeout=60)  # Reduced timeout
                negative_points = neg_future.result(timeout=60)  # Reduced timeout
        else:
            # Sequential processing for small datasets with increased workers
            positive_points = link_points_to_criteria(raw_positive, criteria_list, max_workers + 2)
            negative_points = link_points_to_criteria(raw_negative, criteria_list, max_workers + 2)
        
        # CRITICAL: Apply early sentiment correction based on override flags
        positive_points, negative_points = apply_sentiment_overrides(positive_points, negative_points)
        
        extraction_time = time.time() - extraction_start

        # 4. Collapse multiple points by criteria (optimized)
        collapse_start = time.time()
        pos_collapsed = collapse_points_by_criteria(positive_points)
        neg_collapsed = collapse_points_by_criteria(negative_points)
        
        # 4.5. CRITICAL CORRECTION: Fix misclassified sentiment points
        correction_start = time.time()
        pos_collapsed, neg_collapsed = correct_sentiment_classification(pos_collapsed, neg_collapsed)
        correction_time = time.time() - correction_start
        
        collapse_time = time.time() - collapse_start

        # 5. Calculate weights and percentages with improved scoring system
        calc_start = time.time()
        
        # Use enhanced calculation for better accuracy
        total_positive_weight = sum(
            item.get("criteria_weight", 0.0) 
            for item in pos_collapsed 
            if item.get("criteria_id") is not None
        )
        
        total_negative_weight = sum(
            item.get("criteria_weight", 0.0) 
            for item in neg_collapsed 
            if item.get("criteria_id") is not None
        )
        
        # Calculate percentages based on total weight distribution
        total_weight = total_positive_weight + total_negative_weight
        
        if total_weight > 0:
            # Weight-based percentage calculation
            positive_percentage = (total_positive_weight / total_weight) * 100
            negative_percentage = (total_negative_weight / total_weight) * 100
        else:
            # Fallback to count-based percentage
            pos_count_calc = sum(item.get("count", 0) for item in pos_collapsed)
            neg_count_calc = sum(item.get("count", 0) for item in neg_collapsed)
            total_count_calc = pos_count_calc + neg_count_calc
            
            if total_count_calc > 0:
                positive_percentage = (pos_count_calc / total_count_calc) * 100
                negative_percentage = (neg_count_calc / total_count_calc) * 100
            else:
                positive_percentage = 50.0
                negative_percentage = 50.0
        
        # Calculate gray weight - the remaining weight as decimal (0.0 to 1.0)
        total_gray_weight = max(1.0 - (total_positive_weight + total_negative_weight), 0.0)

        # Enhanced classification label based on percentages
        if total_weight == 0:
            sentiment_label = "neutral"
        elif negative_percentage >= 60:
            sentiment_label = "negative"
        elif positive_percentage >= 60:
            sentiment_label = "positive"
        else:
            sentiment_label = "mixed"
            
        calc_time = time.time() - calc_start

        # 7. Build analysis object with performance metrics and proper structure
        total_processing_time = time.time() - start_time
        
        # Pre-calculate counts for accuracy
        pos_count = sum(item.get("count", 0) for item in pos_collapsed)
        neg_count = sum(item.get("count", 0) for item in neg_collapsed)
        
        analysis_object = {
            "positive_points": pos_collapsed,
            "negative_points": neg_collapsed,
            "classification": sentiment_label,
            "scores": {
                "total_positive_weight": round(total_positive_weight, 3),
                "total_negative_weight": round(total_negative_weight, 3),
                "total_gray_weight": round(total_gray_weight, 3),
                "total_weight": round(total_weight, 3),
                "positive_percentage": f"{round(positive_percentage, 1)}%",
                "negative_percentage": f"{round(negative_percentage, 1)}%"
            },
            "metadata": {
                "total_positive_points": pos_count,
                "total_negative_points": neg_count,
                "total_points": pos_count + neg_count,
                "analysis_timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "extractor_type": "enhanced_feedback_points_extractor_with_sentiment_correction",
                "system_status": {
                    "feedback_extractor_enabled": True,
                    "sentiment_correction_applied": True,
                    "extraction_successful": points_result.error is None,
                    "classification_method": "hybrid_with_intelligent_sentiment_correction"
                },
                "performance": {
                    "total_processing_time": round(total_processing_time, 3),
                    "extraction_time": round(extraction_time, 3),
                    "collapse_time": round(collapse_time, 3),
                    "calculation_time": round(calc_time, 3),
                    "correction_time": round(correction_time, 3),
                    "points_extraction_time": round(points_result.processing_time, 3),
                    "parallel_processing": len(raw_positive) + len(raw_negative) > 4,
                    "extractor_type": "feedback_points_extractor_with_llama"
                }
            }
        }

        # 8. Save analysis result asynchronously (non-blocking)
        def save_async():
            try:
                save_analysis_result(str(client_id), analysis_object)
            except Exception as e:
                logger.error(f"Async save failed: {e}")
        
        # Start save in background thread
        import threading
        save_thread = threading.Thread(target=save_async)
        save_thread.daemon = True
        save_thread.start()

        return {
            "success": True,
            "analysis": analysis_object,
            "positive_score": f"{round(positive_percentage, 1)}%",
            "negative_score": f"{round(negative_percentage, 1)}%",
            "processing_time": round(total_processing_time, 2)
        }
        
    except Exception:
        # Silent failure with basic error info
        error_time = time.time() - start_time
        return {
            "error": "Analysis processing failed",
            "processing_time": round(error_time, 2)
        }
async def cleanup_resources_async():
    """Clean up async resources and connections."""
    try:
        # Save semantic cache before cleanup
        semantic_matcher.save_cache()
        
        # Close async session
        await close_async_session()
        
        # Close MongoDB connection (sync only now)
        if client:
            client.close()
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def cleanup_resources():
    """Clean up global resources and connections."""
    try:
        # Save semantic cache before cleanup
        semantic_matcher.save_cache()
        
        # Close sync MongoDB connection
        if client:
            client.close()
            
        # Run async cleanup
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(cleanup_resources_async())
        except RuntimeError:
            # Event loop is not running, create a new one
            asyncio.run(cleanup_resources_async())
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


async def batch_analyze_transcripts_async(transcripts_data: List[Dict[str, str]], max_workers: int = 6) -> List[Dict[str, Any]]:
    """Ultra-fast async batch analysis with concurrent processing"""
    if not transcripts_data:
        return []
    
    async def analyze_single_async(data):
        try:
            transcript = data.get('transcript', '')[:1200]  # Limit for speed
            client_id = data.get('client_id', '')
            return await analyze_and_calculate_scores_async(transcript, client_id, max_workers=6)
        except Exception as e:
            logger.error(f"Single analysis failed: {e}")
            return {"error": "Analysis failed"}
    
    # Process all transcripts concurrently
    tasks = [analyze_single_async(data) for data in transcripts_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    final_results = []
    for result in results:
        if isinstance(result, Exception):
            final_results.append({"error": "Analysis failed"})
        else:
            final_results.append(result)
    
    return final_results

def batch_analyze_transcripts(transcripts_data: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Analyze multiple transcripts in parallel for maximum efficiency.
    
    Args:
        transcripts_data: List of dicts with 'transcript' and 'client_id' keys
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of analysis results in the same order as input
    """
    if not transcripts_data:
        return []
    
    def analyze_single(data):
        try:
            transcript = data.get('transcript', '')[:1500]  # Limit length for speed
            client_id = data.get('client_id', '')
            return analyze_and_calculate_scores(transcript, client_id, max_workers=4)  # Increased individual workers
        except Exception:
            # Silent failure
            return {"error": "Analysis failed"}
    
    # Increased max workers for better parallelism
    max_workers = min(max_workers + 2, len(transcripts_data), 8)  # Increased max
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(analyze_single, data): i 
            for i, data in enumerate(transcripts_data)
        }
        
        # Collect results in order
        results = [None] * len(transcripts_data)
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result(timeout=100)  # Reduced timeout for speed
                results[index] = result
            except Exception:
                # Silent failure
                results[index] = {"error": "Processing timeout"}
    
    return results


def get_analysis_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for the analysis system."""
    global _criteria_cache
    
    with _cache_lock:
        cache_size = len(_criteria_cache)
        cache_entries = list(_criteria_cache.keys())
    
    # Get LRU cache stats
    cache_info = getattr(_cached_classify_point, 'cache_info', lambda: None)()
    
    stats = {
        "criteria_cache": {
            "size": cache_size,
            "timeout": _cache_timeout,
            "entries": cache_entries[:10]  # First 10 entries
        },
        "classification_cache": {
            "enabled": cache_info is not None
        },
        "database": {
            "connection_pool_size": client.options.pool_options.max_pool_size,
            "min_pool_size": client.options.pool_options.min_pool_size
        },
        "async_session": {
            "active": _async_session is not None and not (_async_session.closed if _async_session else True)
        }
    }
    
    if cache_info:
        stats["classification_cache"].update({
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
            "hit_ratio": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        })
    
    return stats


# Context manager for the entire analysis system
class AnalysisSystem:
    """Enhanced context manager with async support for ultra-fast analysis"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_resources()
    
    def analyze(self, transcript: str, client_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a single transcript using optimized sync method."""
        return analyze_and_calculate_scores(transcript, client_id, **kwargs)
    
    async def analyze_async(self, transcript: str, client_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a single transcript using ultra-fast async method."""
        return await analyze_and_calculate_scores_async(transcript, client_id, **kwargs)
    
    def batch_analyze(self, transcripts_data: List[Dict[str, str]], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple transcripts using optimized sync method."""
        return batch_analyze_transcripts(transcripts_data, **kwargs)
    
    async def batch_analyze_async(self, transcripts_data: List[Dict[str, str]], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple transcripts using ultra-fast async method."""
        return await batch_analyze_transcripts_async(transcripts_data, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return get_analysis_performance_stats()

# Ultra-fast async analysis function for direct use
async def ultra_fast_analyze(transcript: str, client_id: str, max_workers: int = 8) -> Dict[str, Any]:
    """Direct async analysis function with maximum optimization"""
    return await analyze_and_calculate_scores_async(transcript, client_id, max_workers)

# Ultra-fast batch async analysis function
async def ultra_fast_batch_analyze(transcripts_data: List[Dict[str, str]], max_workers: int = 6) -> List[Dict[str, Any]]:
    """Direct async batch analysis with maximum optimization"""
    return await batch_analyze_transcripts_async(transcripts_data, max_workers)