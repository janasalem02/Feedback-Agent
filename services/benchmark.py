"""
أداة قياس أداء موديل LLaMA Storm
Performance Benchmarking Tool for LLaMA Storm Model
"""

import time
import requests
import json
from typing import Dict, List, Any
from datetime import datetime
import statistics
from pathlib import Path

class LLaMAStormBenchmark:
    def __init__(self, model_name: str = "finalend/llama-3.1-storm:8b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.results = []
        
    def run_single_test(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """تشغيل اختبار واحد وقياس الأداء"""
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # حساب الإحصائيات
                tokens_generated = len(response_text.split())
                tokens_per_second = tokens_generated / response_time if response_time > 0 else 0
                
                return {
                    "success": True,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "response_time": response_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                    "response_length": len(response_text),
                    "status_code": response.status_code,
                    "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text
                }
            else:
                return {
                    "success": False,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "response_time": end_time - start_time,
                "error": str(e),
                "status_code": None
            }
    
    def run_benchmark_suite(self) -> Dict[str, Any]:
        """تشغيل مجموعة شاملة من اختبارات الأداء"""
        print(f"🚀 بدء اختبار أداء {self.model_name}")
        print("=" * 60)
        
        # اختبارات متنوعة
        test_prompts = [
            # اختبار سرعة الاستجابة القصيرة
            "ما هو الذكاء الاصطناعي؟",
            
            # اختبار التلخيص
            """لخص هذا النص: تعتبر التكنولوجيا المالية (الفنتك) من أسرع القطاعات نمواً في العالم، 
            حيث تجمع بين التمويل والتكنولوجيا لتقديم خدمات مالية مبتكرة ومريحة للمستخدمين. 
            تشمل هذه الخدمات المدفوعات الرقمية، والإقراض عبر الإنترنت، والاستثمار الآلي، 
            والعملات المشفرة. تهدف شركات الفنتك إلى تحسين تجربة العملاء وتقليل التكاليف 
            وزيادة الوصول إلى الخدمات المالية.""",
            
            # اختبار التحليل
            "حلل إيجابيات وسلبيات العمل عن بُعد",
            
            # اختبار الإبداع
            "اكتب قصة قصيرة عن المستقبل في 50 كلمة",
            
            # اختبار النصائح
            "أعطني 3 نصائح لتحسين الإنتاجية في العمل",
            
            # اختبار الترجمة والفهم
            "Translate to Arabic: Technology is rapidly changing our world"
        ]
        
        print("📊 تشغيل الاختبارات...")
        successful_tests = []
        failed_tests = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 اختبار {i}/{len(test_prompts)}: {prompt[:30]}...")
            result = self.run_single_test(prompt)
            
            if result["success"]:
                successful_tests.append(result)
                print(f"✅ نجح في {result['response_time']:.2f}s - {result['tokens_per_second']:.1f} tokens/s")
            else:
                failed_tests.append(result)
                print(f"❌ فشل: {result.get('error', 'Unknown error')}")
            
            self.results.append(result)
            time.sleep(1)  # فترة راحة قصيرة بين الاختبارات
        
        # حساب الإحصائيات الإجمالية
        if successful_tests:
            response_times = [test["response_time"] for test in successful_tests]
            tokens_per_second = [test["tokens_per_second"] for test in successful_tests]
            
            stats = {
                "total_tests": len(test_prompts),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": (len(successful_tests) / len(test_prompts)) * 100,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "avg_tokens_per_second": statistics.mean(tokens_per_second),
                "min_tokens_per_second": min(tokens_per_second),
                "max_tokens_per_second": max(tokens_per_second),
                "total_tokens_generated": sum(test["tokens_generated"] for test in successful_tests)
            }
        else:
            stats = {
                "total_tests": len(test_prompts),
                "successful_tests": 0,
                "failed_tests": len(failed_tests),
                "success_rate": 0,
                "error": "جميع الاختبارات فشلت"
            }
        
        return {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "detailed_results": self.results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """طباعة النتائج بشكل منسق"""
        print("\n" + "=" * 60)
        print(f"📊 تقرير أداء {results['model_name']}")
        print("=" * 60)
        
        stats = results["statistics"]
        
        if "error" not in stats:
            print(f"🎯 معدل النجاح: {stats['success_rate']:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
            print(f"⏱️  متوسط وقت الاستجابة: {stats['avg_response_time']:.2f}s")
            print(f"📈 أسرع استجابة: {stats['min_response_time']:.2f}s")
            print(f"📉 أبطأ استجابة: {stats['max_response_time']:.2f}s")
            print(f"🔢 الوسيط: {stats['median_response_time']:.2f}s")
            print(f"🚄 متوسط السرعة: {stats['avg_tokens_per_second']:.1f} tokens/second")
            print(f"🔝 أقصى سرعة: {stats['max_tokens_per_second']:.1f} tokens/second")
            print(f"📝 إجمالي الكلمات المولدة: {stats['total_tokens_generated']}")
            
            print(f"\n🏆 تقييم الأداء:")
            if stats['avg_response_time'] < 3:
                print("⚡ ممتاز - استجابة سريعة جداً")
            elif stats['avg_response_time'] < 7:
                print("✅ جيد - استجابة مقبولة")
            else:
                print("⚠️ بطيء - قد تحتاج تحسين")
                
            if stats['avg_tokens_per_second'] > 20:
                print("🚀 سرعة توليد ممتازة")
            elif stats['avg_tokens_per_second'] > 10:
                print("👍 سرعة توليد جيدة")
            else:
                print("🐌 سرعة توليد بطيئة")
        else:
            print(f"❌ {stats['error']}")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """حفظ النتائج في ملف JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 تم حفظ النتائج في: {filename}")

def run_llama_storm_benchmark():
    """تشغيل اختبار أداء LLaMA Storm"""
    benchmark = LLaMAStormBenchmark()
    results = benchmark.run_benchmark_suite()
    benchmark.print_results(results)
    benchmark.save_results(results)
    return results

if __name__ == "__main__":
    run_llama_storm_benchmark()
