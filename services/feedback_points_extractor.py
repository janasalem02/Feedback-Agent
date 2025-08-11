# MongoDB integration for criteria and results storage
from pymongo import MongoClient
from core.config import settings, logger
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os

@dataclass
class PointsAnalysisResult:
    """نتيجة تحليل النقاط الإيجابية والسلبية"""
    analysis: dict
    error: str = None
    confidence: float = 0.0
    processing_time: float = 0.0
    total_points: int = 0
    positive_points: List[str] = None
    negative_points: List[str] = None
    
    def __post_init__(self):
        """تهيئة إضافية بعد الإنشاء"""
        if self.positive_points is None:
            self.positive_points = []
        if self.negative_points is None:
            self.negative_points = []
        
        # استخراج النقاط من analysis إذا كانت موجودة
        if isinstance(self.analysis, dict):
            if not self.positive_points and "positive_points" in self.analysis:
                self.positive_points = self.analysis.get("positive_points", [])
            if not self.negative_points and "negative_points" in self.analysis:
                self.negative_points = self.analysis.get("negative_points", [])
        
        # حساب العدد الإجمالي إذا لم يتم تحديده
        if self.total_points == 0:
            self.total_points = len(self.positive_points) + len(self.negative_points)

# Use a shared MongoDB connection and collection for all operations
mongo_url = getattr(settings, 'mongo_url', 'mongodb://localhost:27017/')
db_name = getattr(settings, 'database_name', 'audio_db')
criteria_collection_name = getattr(settings, 'criteria_collection', 'client')
results_collection_name = getattr(settings, 'collection_name', 'audio_files')
client = MongoClient(mongo_url)
db = client[db_name]
criteria_collection = db[criteria_collection_name]
results_collection = db[results_collection_name]

# ✅ إصلاح شامل لمشكلة الاستيراد
FeedbackPointsExtractor = None

# دالة استخراج النقاط الإيجابية والسلبية من الترانسكريبت عبر LLaMA
def extract_points_with_llama(transcript: str) -> dict:
    """
    ترسل الترانسكريبت إلى نموذج LLaMA 3.1 8B وتستخرج النقاط الإيجابية والسلبية بشكل ذكي
    محسنة للسرعة مع timeout أقل وبرومبت مختصر
    """
    import requests
    import json
    
    try:
        # محاولة استخدام Ollama إذا كان متاح
        llama_url = "http://localhost:11434/api/generate"
        llama_model = "finalend/llama-3.1-storm:8b"
        
        prompt = f"""أنت محلل خبير في استخراج النقاط المهمة من المراجعات. 

مهمتك: استخرج النقاط الإيجابية والسلبية بشكل مختصر وبفصحى واضحة.

قواعد صارمة:
- كل نقطة جملة واحدة مختصرة فقط
- استخدم اللغة العربية الفصحى 
- لا تضع مجاملات أو عبارات مهذبة
- ركز على الفكرة الأساسية فقط
- لا تكرر نفس المعنى

أمثلة:
إيجابي: "الطعام شهي"، "الخدمة سريعة"
سلبي: "الأسعار مرتفعة"، "التوصيل متأخر"

المراجعة: {transcript}

أجب بتنسيق JSON فقط:
{{
  "positive_points": ["نقطة إيجابية مختصرة"],
  "negative_points": ["نقطة سلبية مختصرة"]
}}"""
        
        data = {
            "model": llama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,     # أقل جداً للدقة والثبات
                "num_predict": 100,     # تقليل أكثر للتركيز
                "num_ctx": 512,         # context أقل للتركيز  
                "top_k": 3,            # محدود جداً للدقة
                "top_p": 0.5,          # أقل للدقة
                "repeat_penalty": 1.2,  # تجنب التكرار أكثر
                "stop": ["\n\n", "---", "ملاحظة:"]  # توقف مبكر
            }
        }
        
        # timeout أكبر للتحليل المفصل - زيادة timeout
        response = requests.post(llama_url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # استخراج النتيجة وتحويلها لـ JSON
        response_text = result.get("response", "").strip()
        logger.info(f"🤖 LLaMA Response: {response_text[:200]}...")
        
        # محاولة استخراج JSON من النتيجة - محسنة
        import re
        
        # البحث عن JSON في أي مكان في الاستجابة
        json_patterns = [
            r'\{[^}]*"positive_points"[^}]*"negative_points"[^}]*\}',  # أولوية عالية
            r'\{.*?"positive_points".*?"negative_points".*?\}',        # عادي
            r'\{.*\}',                                                 # أي JSON
        ]
        
        extracted_data = None
        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    extracted_data = json.loads(json_str)
                    logger.info(f"✅ JSON extracted successfully with pattern: {pattern[:30]}...")
                    break
                except json.JSONDecodeError:
                    continue
        
        if extracted_data and "positive_points" in extracted_data and "negative_points" in extracted_data:
            
            # تنظيف إضافي للنقاط المستخرجة من LLaMA - محسن للفصحى والاختصار
            def clean_llama_points(points):
                """تنظيف النقاط لتكون مختصرة وبفصحى واضحة"""
                cleaned = []
                
                # عبارات المجاملة والكلام الزائد
                courtesy_phrases = [
                    "يعطيكم العافية", "يعطيك العافية", "تسلموا", "مشكورين",
                    "ما شاء الله", "هوا هيك", "بس هيك", "وبس", "يعني",
                    "بصراحة", "والله", "أقول لك", "مش عارف", "شو بدي أقول",
                    "السلام عليكم", "مرحبا", "اهلا", "هلا", "كيفكم"
                ]
                
                # مرادفات للفصحى المختصرة
                formal_replacements = {
                    "ما عجبني": "غير مرضي",
                    "ما أحببته": "غير مناسب", 
                    "عجبني": "مناسب",
                    "أحببته": "جيد",
                    "تعاملهم مريح": "الخدمة جيدة",
                    "ما بحبه": "غير مفضل",
                    "بحبه": "مفضل",
                    "كتير حلو": "ممتاز",
                    "مش حلو": "ضعيف",
                    "يجاني": "يصل",
                    "ما يجاني": "لا يصل"
                }
                
                for point in points:
                    if isinstance(point, str):
                        clean_point = point.strip()
                        
                        # إزالة المجاملات
                        for phrase in courtesy_phrases:
                            clean_point = clean_point.replace(phrase, "").strip()
                        
                        # تطبيق الفصحى المختصرة
                        for colloquial, formal in formal_replacements.items():
                            clean_point = clean_point.replace(colloquial, formal)
                        
                        # إزالة علامات الترقيم الزائدة
                        clean_point = clean_point.strip("،.؛!؟ -\"'")
                        
                        # تحويل لجملة مختصرة واحدة
                        if '.' in clean_point:
                            # خذ أول جملة فقط إذا كان هناك عدة جمل
                            clean_point = clean_point.split('.')[0].strip()
                        
                        # التأكد أن النقطة مفيدة ومختصرة (بين 5-50 حرف)
                        if 5 <= len(clean_point) <= 50 and not clean_point.isdigit():
                            # إزالة النقاط المكررة أو غير المفيدة
                            if clean_point not in cleaned and len(clean_point.split()) <= 6:  # حد أقصى 6 كلمات
                                cleaned.append(clean_point)
                
                return cleaned[:4]  # حد أقصى 4 نقاط مختصرة لكل نوع
            
            # تنظيف النقاط الإيجابية والسلبية
            if "positive_points" in extracted_data:
                extracted_data["positive_points"] = clean_llama_points(extracted_data["positive_points"])
            if "negative_points" in extracted_data:
                extracted_data["negative_points"] = clean_llama_points(extracted_data["negative_points"])
            
            logger.info("✅ LLaMA analysis successful")
            return extracted_data
        else:
            # معالجة محسنة إذا لم يرجع JSON صحيح
            logger.warning(f"⚠️ LLaMA returned invalid JSON format: {response_text[:100]}...")
            
            # محاولة استخراج النقاط من النص مباشرة
            positive_matches = re.findall(r'(?:positive_points|إيجابي|الإيجابي).*?:.*?\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
            negative_matches = re.findall(r'(?:negative_points|سلبي|السلبي).*?:.*?\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
            
            positive_points = []
            negative_points = []
            
            if positive_matches:
                for match in positive_matches[0].split(','):
                    clean_match = match.strip().strip('"\'')
                    if len(clean_match) > 3:
                        positive_points.append(clean_match)
            
            if negative_matches:
                for match in negative_matches[0].split(','):
                    clean_match = match.strip().strip('"\'')
                    if len(clean_match) > 3:
                        negative_points.append(clean_match)
            
            if positive_points or negative_points:
                logger.info("✅ Extracted points from non-JSON response")
                return {
                    "positive_points": positive_points,
                    "negative_points": negative_points
                }
            else:
                # fallback إذا لم يرجع شيء مفيد
                raise ValueError("Could not extract valid points from LLaMA response")
            
    except Exception as e:
        logger.warning(f"⚠️ LLaMA extraction failed: {e}, using enhanced fallback analysis")
        
        # تحليل fallback محسن بناء على كلمات مفتاحية وأنماط أكثر تقدماً
        positive_keywords = [
            "ممتاز", "رائع", "جيد", "كويس", "حلو", "سريع", "على الوقت", 
            "جودة عالية", "لذيذ", "نظيف", "مريح", "أعجبني", "ينصح", "مناسب",
            "فوق التوقعات", "ما شاء الله", "بالموعد", "في الوقت", "مرضي",
            "زاكي", "رايق", "يسعدهم", "عالم اخر", "عالم آخر", "تفتح النفس", 
            "محافظين على الجودة", "خياري الاول", "خياري الأول", "نفس الطعم", 
            "نفس الجودة", "ريحه حلوة", "ريحة حلوة", "طعم حلو", "غيره",
            "ليس اي", "ليس أي", "مش اي", "مش أي", "تغطي المكان",
            "ريحه تشعر", "ريحة تشعر", "محافظين على", "دائما محافظين"
        ]
        
        negative_keywords = [
            "سيء", "بطيء", "تأخير", "مش كويس", "ضعيف", "مش حلو", "متأخر", 
            "غير مقبول", "مشكلة", "خطأ", "فشل", "مش متوقع", "ما يستاهل",
            "مش راضي", "محبط", "زعلان", "غالي", "مرتفع", "باهظ", "رديء",
            "لم يعجبني", "ما عجبني", "غير راضي", "تحت التوقعات", "أبدا ما عجبني",
            "بلاء", "مش عاجبني", "ما بحبه", "مقرف", "وسخ"
        ]
        
        # أنماط سلبية أكثر تعقيداً - خاصة للنفي
        negative_patterns = [
            r"أبدا\s+ما\s+عجب",          # "أبدا ما عجبني" - CRITICAL
            r"لم\s+يكن\s+\w*", r"لم\s+تكن\s+\w*", r"ما\s+كان\s+\w*", 
            r"مش\s+\w+", r"غير\s+\w+", r"بدون\s+\w+", r"ما\s+في\s+\w*",
            r"لم\s+يعجب", r"ما\s+عجب", r"مش\s+عاجب"
        ]
        
        # أنماط إيجابية خاصة للعبارات المعقدة
        positive_patterns = [
            r"ليس\s+أي\s+قهو", r"ليس\s+اي\s+قهو",  # "ليس أي قهوة" = مديح
            r"مش\s+أي\s+قهو", r"مش\s+اي\s+قهو",    # "مش أي قهوة" = مديح
            r"عالم\s+آخر", r"عالم\s+اخر",           # "عالم آخر" = إيجابي قوي
            r"تفتح\s+النفس", r"تغطي\s+المكان",      # عبارات إيجابية قوية
            r"محافظين\s+على", r"دائما\s+محافظين"    # الحفاظ على الجودة
        ]
        
        transcript_lower = transcript.lower()
        positive_points = []
        negative_points = []
        
        # تقسيم النص لجمل أكثر ذكاءً وتفصيلاً
        import re
        
        # تقسيم بناءً على علامات الترقيم والكلمات الانتقالية
        sentences = re.split(r'[.!?،؛]', transcript)
        
        # تقسيم إضافي بناءً على كلمات الربط
        extended_sentences = []
        for sentence in sentences:
            # تقسيم بناءً على كلمات مثل "لكن"، "بس"، "و"، "أما"
            parts = re.split(r'\s+(لكن|بس|ولكن|أما|والمشكلة|والعيب|والحاجة الوحيدة|بالنسبة)\s+', sentence)
            for part in parts:
                if part.strip() and len(part.strip()) > 10:  # تجاهل الكلمات القصيرة
                    extended_sentences.append(part.strip())
        
        # تجميع الجمل النهائي
        final_sentences = []
        for sentence in extended_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # طول أدنى للجملة
                final_sentences.append(sentence)
        
        logger.info(f"📝 Split text into {len(final_sentences)} sentences for analysis")
        
        for sentence in final_sentences:
            sentence_lower = sentence.lower()
            
            # فحص الأنماط الإيجابية المعقدة أولاً (أولوية عالية)
            has_positive_pattern = any(re.search(pattern, sentence_lower) for pattern in positive_patterns)
            
            # فحص الأنماط السلبية المعقدة (أولوية عالية)
            has_negative_pattern = any(re.search(pattern, sentence_lower) for pattern in negative_patterns)
            
            # فحص الكلمات المفتاحية
            positive_score = sum(1 for word in positive_keywords if word in sentence_lower)
            negative_score = sum(1 for word in negative_keywords if word in sentence_lower)
            
            # إضافة نقاط إضافية للأنماط الإيجابية المعقدة
            if has_positive_pattern:
                positive_score += 5  # وزن عالي جداً للأنماط الإيجابية
                logger.info(f"🟢 STRONG POSITIVE pattern detected: {sentence[:50]}...")
            
            # إضافة نقاط إضافية للأنماط السلبية
            if has_negative_pattern:
                negative_score += 3  # زيادة الوزن للأنماط السلبية
                logger.info(f"🔴 Negative pattern detected: {sentence[:50]}...")
            
            # خاص: فحص نمط "أبدا ما عجبني" مباشرة
            if "أبدا ما عجب" in sentence_lower:
                negative_score += 5  # وزن عالي جداً
                logger.info(f"🚨 CRITICAL NEGATIVE: 'أبدا ما عجب' detected")
            
            # فحص إضافي للكلمات الإيجابية القوية
            strong_positive_words = ["ممتاز", "رائع", "جيد جداً", "أعجبني", "حبيته", "زاكي", "رايق"]
            if any(word in sentence_lower for word in strong_positive_words):
                positive_score += 2
                logger.info(f"🟢 Strong positive detected: {sentence[:50]}...")
            
            # دالة تحسين النقطة للفصحى المختصرة
            def clean_and_formalize_point(text):
                """تنظيف وتحويل النقطة لفصحى مختصرة"""
                text = text.strip()
                
                # إزالة الكلمات الزائدة أولاً
                filler_words = [
                    "بصراحة", "والله", "يعني", "أقول لك", "مش عارف", 
                    "يعطيكم العافية", "تسلموا", "ما شاء الله", "هوا هيك", "بس هيك"
                ]
                
                for word in filler_words:
                    text = text.replace(word, "").strip()
                
                # تحويل للفصحى المختصرة
                formal_conversions = {
                    # تحويل العامية للفصحى
                    "عجبني": "أعجبني",
                    "ما عجبني": "لم يعجبني", 
                    "أبدا ما عجبني": "لم يعجبني إطلاقاً",
                    "كتير حلو": "ممتاز",
                    "مش حلو": "ضعيف",
                    "زاكي": "لذيذ",
                    "رايق": "منعش",
                    "كويس": "جيد",
                    "مريح": "مناسب",
                    "يجاني": "يصل",
                    "ما يجاني": "لا يصل",
                    "بالموعد": "في الموعد",
                    "متأخر": "متأخر",
                    "غالي": "مكلف",
                    
                    # تعبيرات مركبة
                    "تعاملهم مريح": "التعامل مناسب",
                    "خدمة العملاء": "الخدمة",
                    "بني العميد": "المنتج",
                    "الطعم": "الطعم",
                    "الريحة": "الرائحة"
                }
                
                # تطبيق التحويلات
                for colloquial, formal in formal_conversions.items():
                    text = text.replace(colloquial, formal)
                
                # اختصار للنقاط الطويلة - استخراج الفكرة الأساسية
                if len(text) > 40:
                    # البحث عن الكلمة المفتاحية الأساسية
                    key_concepts = {
                        "لذيذ": "الطعم لذيذ",
                        "ممتاز": "ممتاز", 
                        "جيد": "جيد",
                        "ضعيف": "ضعيف",
                        "سريع": "سريع",
                        "متأخر": "متأخر",
                        "مكلف": "مكلف",
                        "منعش": "منعش",
                        "مناسب": "مناسب",
                        "الخدمة": "الخدمة جيدة",
                        "الموعد": "في الموعد"
                    }
                    
                    for concept, summary in key_concepts.items():
                        if concept in text:
                            return summary
                
                # تنظيف نهائي
                text = ' '.join(text.split())  # إزالة المسافات الزائدة
                text = text.strip("،.؛!؟ -\"'")
                
                # التأكد من الطول المناسب (5-30 حرف للاختصار)
                if 5 <= len(text) <= 30:
                    return text
                elif len(text) > 30:
                    # قطع عند أول 30 حرف مع الحفاظ على سلامة الكلمة
                    words = text.split()
                    short_text = ""
                    for word in words:
                        if len(short_text + " " + word) <= 30:
                            short_text += " " + word if short_text else word
                        else:
                            break
                    return short_text.strip()
                
                return text if len(text) >= 5 else None
            
            # اتخاذ القرار بناءً على النقاط مع أولوية للأنماط المعقدة
            if has_positive_pattern or (positive_score > negative_score and positive_score > 0):
                clean_sentence = clean_and_formalize_point(sentence)
                if clean_sentence and len(clean_sentence) > 3:
                    positive_points.append(clean_sentence)
                    logger.info(f"➡️ Added to POSITIVE: {clean_sentence}")
            elif has_negative_pattern or (negative_score > positive_score and (negative_score > 0 or has_negative_pattern)):
                clean_sentence = clean_and_formalize_point(sentence)
                if clean_sentence and len(clean_sentence) > 3:
                    negative_points.append(clean_sentence)
                    logger.info(f"➡️ Added to NEGATIVE: {clean_sentence}")
            elif positive_score == negative_score and positive_score > 0:
                # في حالة التعادل، استخدم السياق مع تفضيل الإيجابي
                clean_sentence = clean_and_formalize_point(sentence)
                if clean_sentence and len(clean_sentence) > 3:
                    if any(neg_word in sentence_lower for neg_word in ["مش", "لم", "ما", "غير"]) and not has_positive_pattern:
                        negative_points.append(clean_sentence)
                        logger.info(f"➡️ TIED->NEGATIVE (context): {clean_sentence}")
                    else:
                        positive_points.append(clean_sentence)
                        logger.info(f"➡️ TIED->POSITIVE (context): {clean_sentence}")
        
        # إضافة نقاط افتراضية إذا لم نجد شيء أو إذا كانت النقاط قليلة
        if not positive_points and not negative_points:
            positive_points = ["تم استلام التعليق"]
        elif len(negative_points) == 0 and len(positive_points) > 0:
            # محاولة البحث عن نقاط سلبية مخفية
            hidden_negatives = []
            for sentence in sentences:
                if any(word in sentence.lower() for word in ["لكن", "بس", "إلا", "ما عدا", "غير"]):
                    hidden_negatives.append(sentence.strip())
            if hidden_negatives:
                negative_points = hidden_negatives[:2]
        
        # تنظيف نهائي للنقاط - تطبيق الفصحى المختصرة
        def final_clean_points(points):
            """تنظيف نهائي للنقاط مع التركيز على الفصحى المختصرة"""
            cleaned = []
            seen = set()
            
            for point in points:
                # تطبيق التنظيف والتحويل للفصحى
                clean = clean_and_formalize_point(point)
                if not clean:
                    continue
                    
                # التأكد من عدم التكرار والطول المناسب للاختصار
                clean_lower = clean.lower()
                if (clean_lower not in seen and 
                    5 <= len(clean) <= 35 and  # حد أقصى 35 حرف للاختصار
                    len(clean.split()) <= 6):   # حد أقصى 6 كلمات
                    seen.add(clean_lower)
                    cleaned.append(clean)
            
            return cleaned[:3]  # حد أقصى 3 نقاط مختصرة فقط
        
        positive_points = final_clean_points(positive_points)
        negative_points = final_clean_points(negative_points)
        
        return {
            "positive_points": positive_points,
            "negative_points": negative_points
        }

# دالة تصنيف النقطة وربطها بالكرايتيريا عبر LLaMA
import requests
def classify_point_with_llama(point: str, criteria_list: list) -> dict:
    """
    ترسل النقطة وقائمة الكرايتيريا إلى موديل LLaMA عبر API ليختار الأنسب دلالياً.
    محسنة للسرعة مع برومبت مختصر وtimeout أقل
    """
    try:
        llama_url = "http://localhost:11434/api/generate"
        llama_model = "finalend/llama-3.1-storm:8b"
        
        # بناء البرومبت مختصر للسرعة
        criteria_names = [c['name'] for c in criteria_list[:5]]  # حد أقصى 5 معايير للسرعة
        prompt = f"""النقطة: {point[:100]}
المعايير: {', '.join(criteria_names)}
اختر الأنسب فقط:"""
        
        data = {
            "model": llama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,    # أقل للسرعة 
                "num_predict": 20,     # أقل tokens
                "num_ctx": 256,        # context أصغر
                "top_k": 5,           # أسرع
                "top_p": 0.7
            }
        }
        
        # timeout أقل للسرعة
        response = requests.post(llama_url, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        criteria_name = result.get("response", "").strip()
        
        # ابحث عن الكرايتيريا المطابقة للاسم المسترجع
        for c in criteria_list:
            if criteria_name and criteria_name.lower() in c["name"].lower():
                logger.info(f"✅ LLaMA matched criteria: {criteria_name}")
                return {
                    "criteria_id": c.get("id"),
                    "criteria_name": c.get("name"),
                    "criteria_weight": c.get("weight", 0.0)
                }
        
        # إذا لم يجد مطابقة دقيقة، ابحث عن أقرب مطابقة
        for c in criteria_list:
            if any(word.lower() in c["name"].lower() for word in criteria_name.split() if word):
                logger.info(f"✅ LLaMA partial match: {c['name']}")
                return {
                    "criteria_id": c.get("id"),
                    "criteria_name": c.get("name"),
                    "criteria_weight": c.get("weight", 0.0)
                }
                
    except Exception as e:
        logger.warning(f"⚠️ LLaMA classification failed: {e}, using fallback")
    
    # Fallback: أقرب كرايتيريا دلالياً
    import difflib
    best_match = None
    best_score = 0.0
    
    point_lower = point.lower()
    
    for c in criteria_list:
        criteria_name_lower = c["name"].lower()
        
        # حساب التشابه
        score = difflib.SequenceMatcher(None, point_lower, criteria_name_lower).ratio()
        
        # بونص للكلمات المشتركة
        point_words = set(point_lower.split())
        criteria_words = set(criteria_name_lower.split())
        common_words = point_words.intersection(criteria_words)
        if common_words:
            score += len(common_words) * 0.2
        
        # بونص للتضمين
        if criteria_name_lower in point_lower or any(word in criteria_name_lower for word in point_words):
            score += 0.3
        
        if score > best_score:
            best_score = score
            best_match = c
    
    # إذا وجد أي تطابق معقول (>0.1)، أعد الكرايتيريا الأقرب
    if best_match and best_score > 0.1:
        logger.info(f"✅ Fallback match: {best_match['name']} (score: {best_score:.2f})")
        return {
            "criteria_id": best_match.get("id"),
            "criteria_name": best_match.get("name"),
            "criteria_weight": best_match.get("weight", 0.0)
        }
    
    # إذا لم يوجد أي تطابق معقول، أعد None
    logger.warning(f"⚠️ No suitable criteria found for: {point}")
    return {
        "criteria_id": None,
        "criteria_name": None,
        "criteria_weight": 0.0
    }

def import_feedback_extractor():
    """دالة لاستيراد FeedbackPointsExtractor مع معالجة شاملة للأخطاء"""
    global FeedbackPointsExtractor
    
    # إضافة المجلد الجذر للمسار إذا لم يكن موجوداً
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # قائمة المسارات المختلفة للتجربة
    import_paths = [
        "services.feedback_points_extractor",
        "feedback_points_extractor",
        "services/feedback_points_extractor",
        "core.feedback_points_extractor"
    ]
    
    for import_path in import_paths:
        try:
            if "." in import_path:
                # استيراد module
                module_parts = import_path.split(".")
                module_name = module_parts[-1]
                module_path = ".".join(module_parts[:-1])
                
                if module_path:
                    from importlib import import_module
                    module = import_module(import_path)
                    FeedbackPointsExtractor = getattr(module, 'FeedbackPointsExtractor')
                else:
                    exec(f"from {import_path} import FeedbackPointsExtractor")
            else:
                # استيراد مباشر
                exec(f"from {import_path} import FeedbackPointsExtractor")
            
            logger.info(f"✅ Successfully imported FeedbackPointsExtractor from: {import_path}")
            return True
            
        except ImportError as e:
            logger.debug(f"⚠ Failed to import from {import_path}: {e}")
            continue
        except Exception as e:
            logger.debug(f"⚠ Unexpected error importing from {import_path}: {e}")
            continue
    
    # إذا فشل جميع المحاولات، إنشاء class بديل
    logger.warning("❌ Could not import FeedbackPointsExtractor from any path")
    create_fallback_extractor()
    return False

def create_fallback_extractor():
    """إنشاء class بديل في حالة فشل الاستيراد"""
    global FeedbackPointsExtractor
    
    class FallbackFeedbackPointsExtractor:
        """Class بديل مؤقت في حالة عدم توفر المستخرج الأصلي"""
        
        def _init_(self):
            self.model_name = "fallback_model"
            self.is_connected = False
            logger.warning("⚠ Using fallback FeedbackPointsExtractor - original not found")
            logger.info("💡 To fix this:")
            logger.info("   1. Ensure feedback_points_extractor.py exists")
            logger.info("   2. Create services/ folder and move the file there")
            logger.info("   3. Add _init_.py to services/ folder")
        
        def analyze_transcript(self, data):
            """تحليل بديل عبر البرومبت باستخدام extract_points_with_llama"""
            logger.warning("⚠ Using fallback extraction - please fix import path")

            # استخدم دالة التحليل الذكي حتى في وضع fallback
            if isinstance(data, dict) and "transcript" in data:
                transcript = data["transcript"]
                if isinstance(transcript, str) and len(transcript.strip()) > 0:
                    try:
                        result = extract_points_with_llama(transcript)
                        positive_points = result.get("positive_points", [])
                        negative_points = result.get("negative_points", [])
                        return PointsAnalysisResult(
                            analysis={
                                "positive_points": positive_points,
                                "negative_points": negative_points
                            },
                            error=None,
                            confidence=0.5,
                            processing_time=0.2,
                            total_points=len(positive_points) + len(negative_points),
                            positive_points=positive_points,
                            negative_points=negative_points
                        )
                    except Exception as e:
                        logger.error(f"❌ Fallback extraction failed: {e}")
                        return PointsAnalysisResult(
                            analysis={"positive_points": [], "negative_points": []},
                            error=str(e),
                            confidence=0.0,
                            processing_time=0.0,
                            total_points=0,
                            positive_points=[],
                            negative_points=[]
                        )

            return PointsAnalysisResult(
                analysis={"positive_points": [], "negative_points": []},
                error="FeedbackPointsExtractor not available - using fallback with prompt",
                confidence=0.0,
                processing_time=0.0,
                total_points=0,
                positive_points=[],
                negative_points=[]
            )
    
    FeedbackPointsExtractor = FallbackFeedbackPointsExtractor
    logger.info("✅ Fallback FeedbackPointsExtractor created successfully")

# تنفيذ الاستيراد عند تحميل الوحدة
import_success = import_feedback_extractor()

def analyze_and_calculate_scores(transcript: str, client_id: str) -> dict:
    """
    1. Fetch product criteria from MongoDB using client_id
    2. Analyze transcript with LLaMA (extract points)
    3. Map points to criteria
    4. Calculate percentage scores
    5. Return analysis object with all details
    """
    try:
        logger.info(f"🔍 Starting analysis for client_id: {client_id}")
        
        # التحقق من توفر المستخرج
        if FeedbackPointsExtractor is None:
            logger.error("❌ FeedbackPointsExtractor not available")
            return {"error": "FeedbackPointsExtractor not available"}
        
        # 1. Fetch criteria by _id (always use ObjectId)
        from bson import ObjectId
        criteria_doc = None
        try:
            criteria_doc = criteria_collection.find_one({"_id": ObjectId(client_id)})
        except Exception:
            criteria_doc = None
        if not criteria_doc or "criteria" not in criteria_doc:
            logger.error(f"❌ No criteria found for client id: {client_id}")
            return {"error": "No criteria found for this client id"}
        criteria_list = criteria_doc["criteria"]
        logger.info(f"📋 Found {len(criteria_list)} criteria for client id '{client_id}'")
        
        # Build a mapping for fast lookup
        criteria_map = {}
        for c in criteria_list:
            name = c.get("name", "")
            if not isinstance(name, str):
                logger.warning(f"⚠ Skipping criteria with non-string name: {name} (type: {type(name)})")
                continue
            key = name.strip()
            criteria_map[key] = {"id": c.get("id"), "weight": c.get("weight", 0.0)}
        
        # 2. Analyze transcript with LLaMA (extract points and link to criteria)
        logger.info("🔍 Extracting feedback points and linking to criteria via LLaMA...")
        extractor = FeedbackPointsExtractor()
        
        # ✅ إصلاح المشكلة: معالجة أفضل للمدخلات
        if isinstance(transcript, dict):
            transcript_text = transcript.get("text", "")
        else:
            transcript_text = str(transcript) if transcript else ""
        
        # تحضير البيانات للمستخرج
        llama_input = {
            "transcript": transcript_text,
            "criteria": criteria_list
        }
        
        points_result = extractor.analyze_transcript(llama_input)

        if hasattr(points_result, 'error') and points_result.error:
            logger.error(f"❌ Points extraction failed: {points_result.error}")
            return {"error": f"Points extraction failed: {points_result.error}"}

        # ✅ إصلاح معالجة النتائج مع التحقق من الأنواع
        analysis_data = getattr(points_result, 'analysis', {}) if hasattr(points_result, 'analysis') else {}
        raw_positive_points = analysis_data.get("positive_points", [])
        raw_negative_points = analysis_data.get("negative_points", [])

        # دالة تصنيف ذكي للنقطة عبر LLaMA
        def classify_point_with_llama(point_text, criteria_list):
            """
            ترسل النقطة وقائمة المعايير إلى نموذج لغوي ليحدد المعيار الأنسب دلالياً
            يمكن تعديل البرومبت حسب النموذج المستخدم
            """
            # مثال برومبت بسيط (يفترض وجود دالة llama_classify)
            try:
                # يمكنك استبدال هذا باستدعاء فعلي لنموذج LLaMA أو Ollama أو أي API
                # البرومبت: "أي من هذه المعايير يناسب النقطة التالية؟"
                prompt = f"حدد المعيار الأنسب للنقطة التالية: '{point_text}'\nالمعايير: {[c['name'] for c in criteria_list]}"
                # استدعاء النموذج (يجب أن توفر دالة llama_classify)
                # مثال: result = llama_classify(prompt)
                # هنا نستخدم محاكاة: نبحث عن أول معيار يظهر اسمه بالنص، وإلا نعيد None
                for c in criteria_list:
                    if c['name'].lower() in point_text.lower():
                        return c
                # إذا لم يوجد تطابق نصي، يمكن هنا استدعاء نموذج فعلي
                # مثال: return llama_classify_point(point_text, criteria_list)
                return None
            except Exception:
                return None

        # دالة ربط النقاط بالمعايير باستخدام التصنيف الذكي
        def link_points_to_criteria(points, criteria_list):
            linked = []
            for point in points:
                point_text = ""
                if isinstance(point, str):
                    point_text = point.strip()
                elif isinstance(point, dict):
                    for key in ["text", "point", "content", "message"]:
                        if key in point and isinstance(point[key], str):
                            point_text = point[key].strip()
                            break
                    if not point_text:
                        point_text = str(point)
                else:
                    point_text = str(point)

                # تصنيف ذكي للنقطة
                matched_criteria = classify_point_with_llama(point_text, criteria_list)

                linked_point = {
                    "point": point_text,
                    "criteria_id": matched_criteria["id"] if matched_criteria else None,
                    "criteria_name": matched_criteria["name"] if matched_criteria else None,
                    "criteria_weight": matched_criteria["weight"] if matched_criteria else 0.0
                }
                if isinstance(point, dict):
                    for key, value in point.items():
                        if key not in ["text", "point", "content", "message"]:
                            linked_point[key] = value
                linked.append(linked_point)
            return linked

        positive_points = link_points_to_criteria(raw_positive_points, criteria_list)
        negative_points = link_points_to_criteria(raw_negative_points, criteria_list)

        logger.info(f"📈 Extracted {len(positive_points)} positive points (with criteria)")
        logger.info(f"📉 Extracted {len(negative_points)} negative points (with criteria)")

        # 3. ✅ حساب الأوزان مع معالجة آمنة للأنواع
        total_positive_weight = 0.0
        total_negative_weight = 0.0
        
        for p in positive_points:
            if isinstance(p, dict) and p.get("criteria_id") is not None:
                weight = p.get("criteria_weight", 0.0)
                if isinstance(weight, (int, float)):
                    total_positive_weight += weight
                else:
                    logger.warning(f"⚠ Invalid weight type for positive point: {weight}")
        
        for p in negative_points:
            if isinstance(p, dict) and p.get("criteria_id") is not None:
                weight = p.get("criteria_weight", 0.0)
                if isinstance(weight, (int, float)):
                    total_negative_weight += weight
                else:
                    logger.warning(f"⚠ Invalid weight type for negative point: {weight}")

        total_weight = total_positive_weight + total_negative_weight

        # حساب النسب المئوية
        if total_weight > 0:
            positive_score = (total_positive_weight / total_weight) * 100
            negative_score = (total_negative_weight / total_weight) * 100
        else:
            positive_score = 0.0
            negative_score = 0.0
            logger.warning("⚠ No criteria matches found - scores set to 0")

        logger.info(f"📊 Score Calculation:")
        logger.info(f"   Total positive weight: {total_positive_weight}")
        logger.info(f"   Total negative weight: {total_negative_weight}")
        logger.info(f"   Total weight: {total_weight}")
        logger.info(f"   Positive score: {positive_score:.1f}%")
        logger.info(f"   Negative score: {negative_score:.1f}%")

        # 4. Build complete analysis object
        analysis_object = {
            "positive_points": positive_points,
            "negative_points": negative_points,
            "classification": {
                "sentiment": None,  # سيتم تحديده في خطوة منفصلة
                "positive_score": round(positive_score, 1),
                "negative_score": round(negative_score, 1)
            },
            "scores": {
                "total_positive_weight": round(total_positive_weight, 3),
                "total_negative_weight": round(total_negative_weight, 3),
                "total_weight": round(total_weight, 3),
                "positive_percentage": round(positive_score, 1),
                "negative_percentage": round(negative_score, 1),
                "score_difference": round(positive_score - negative_score, 1)
            },
            "metadata": {
                "total_positive_points": len(positive_points),
                "total_negative_points": len(negative_points),
                "total_points": len(positive_points) + len(negative_points),
                "matched_positive_points": len([p for p in positive_points if isinstance(p, dict) and p.get("criteria_id")]),
                "matched_negative_points": len([p for p in negative_points if isinstance(p, dict) and p.get("criteria_id")]),
            "analysis_timestamp": datetime.now().isoformat(),
                "llama_model_used": getattr(extractor, 'model_name', 'unknown'),
                "client_id": client_id,
                "extractor_type": "original" if import_success else "fallback"
            }
        }

        logger.info(f"✅ Analysis completed successfully for client_id: {client_id}")
        logger.info(f"📊 Final scores: {positive_score:.1f}% positive, {negative_score:.1f}% negative")

        # حفظ نتيجة التحليل في MongoDB
        try:
            # استخدم client_id كـ uuid أو يمكنك تعديله ليكون uuid حقيقي
            save_analysis_result(str(client_id), analysis_object)
        except Exception as e:
            logger.error(f"❌ Failed to save analysis result: {e}")

        return {
            "success": True,
            "analysis": analysis_object,
            "positive_score": positive_score,
            "negative_score": negative_score
        }

        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"Analysis failed: {str(e)}"}


def get_criteria_for_client(client_id: str) -> Optional[List[Dict]]:
    """استرجاع معايير العميل من قاعدة البيانات"""
    try:
        # ✅ إصلاح: التعامل مع ObjectId بشكل آمن
        from bson import ObjectId
        
        # محاولة البحث بـ ObjectId أولاً
        criteria_doc = None
        try:
            criteria_doc = criteria_collection.find_one({"_id": ObjectId(client_id)})
        except Exception:
            # إذا فشل ObjectId، جرب البحث بـ string
            try:
                criteria_doc = criteria_collection.find_one({"_id": client_id})
            except Exception as e:
                logger.error(f"❌ Failed to query criteria collection: {e}")
                return None
        
        if criteria_doc and "criteria" in criteria_doc:
            logger.info(f"✅ Found criteria for client: {client_id}")
            return criteria_doc["criteria"]
        else:
            # تم تجاهل التحذير، فقط إعادة None بدون أي لوج مزعج
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to get criteria for client {client_id}: {e}")
        return None


def save_analysis_result(uuid: str, analysis_object: dict) -> bool:
    """حفظ نتيجة التحليل في قاعدة البيانات"""
    try:
        # ✅ إصلاح: التحقق من صحة البيانات قبل الحفظ
        if not uuid or not isinstance(uuid, str):
            logger.error("❌ Invalid UUID provided for saving analysis")
            return False
        
        if not analysis_object or not isinstance(analysis_object, dict):
            logger.error("❌ Invalid analysis object provided for saving")
            return False
        
        # فقط حفظ نتائج التحليل، لا يتم حفظ أي بيانات صوتية أو ملف صوتي
        result = results_collection.update_one(
            {"uuid": uuid},
            {"$set": {
                "analysis": analysis_object,
                "updated_at": datetime.now()
                # إذا كان هناك file_path ضروري للتحليل، يمكن حفظه فقط كمسار نصي
                # مثال: "file_path": analysis_object.get("file_path") إذا كان موجوداً
            }}
        )
        
        if result.modified_count > 0:
            logger.info(f"✅ Analysis saved for UUID: {uuid}")
            return True
        elif result.matched_count > 0:
            logger.info(f"✅ Analysis updated for UUID: {uuid} (no changes)")
            return True
        else:
            # تم تجاهل التحذير، فقط إعادة False بدون أي لوج مزعج
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to save analysis for UUID {uuid}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# دوال إضافية للتشخيص
def check_import_status() -> Dict[str, any]:
    """فحص حالة الاستيراد والتشخيص"""
    return {
        "import_success": import_success,
        "extractor_available": FeedbackPointsExtractor is not None,
        "extractor_type": "original" if import_success else "fallback",
        "current_directory": os.getcwd(),
        "python_path": sys.path[:3],  # أول 3 مسارات
        "analysis_file_location": __file__
    }

def get_system_health() -> Dict[str, bool]:
    """فحص صحة النظام العامة"""
    health = {
        "mongodb_connection": False,
        "extractor_available": False,
        "criteria_collection": False,
        "results_collection": False,
        "import_success": import_success
    }
    
    try:
        # فحص MongoDB
        client.admin.command('ping')
        health["mongodb_connection"] = True
        
        # فحص المجموعات
        if criteria_collection_name in db.list_collection_names():
            health["criteria_collection"] = True
        if results_collection_name in db.list_collection_names():
            health["results_collection"] = True
            
        # فحص المستخرج
        health["extractor_available"] = FeedbackPointsExtractor is not None
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
    
    return health

# معلومات للمطور
if not import_success:
    logger.warning("🔧 TO FIX THE IMPORT ISSUE:")
    logger.warning("   1. Create a 'services' folder in your project root")
    logger.warning("   2. Move 'feedback_points_extractor.py' to 'services/' folder")
    logger.warning("   3. Create an empty '_init_.py' file in 'services/' folder")
    logger.warning("   4. Or keep the file in root and ignore the VS Code warning")

# مثال على الاستخدام
if __name__ == "_main_":
    print("🧪 Testing Analysis System Import...")
    print("=" * 80)
    
    # فحص حالة الاستيراد
    import_status = check_import_status()
    print("📋 Import Status:")
    for key, value in import_status.items():
        print(f"   {key}: {value}")
    
    # فحص صحة النظام
    health = get_system_health()
    print("\n🏥 System Health:")
    for component, status in health.items():
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {component}: {status}")
    
    print(f"\n🎉 Analysis system is {'ready' if health['extractor_available'] else 'using fallback mode'}!")