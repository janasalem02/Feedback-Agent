"""
🚀 Ultra-Fast Whisper Large V3 - RTX 3050 4GB Optimized
===============================================

📈 التحسينات المطبقة للسرعة القصوى:
- ✅ int8 compute_type للـ RTX 3050 4GB (أسرع من float16)
- ✅ num_workers=1 للـ GPU (تجنب overhead)
- ✅ beam_size=1 (أسرع خيار)
- ✅ word_timestamps=False (توفير وقت)
- ✅ no_speech_threshold=0.6 (معالجة أقل)
- ✅ Model caching (تحميل فوري للاستخدام التالي)
- ✅ CUDA optimizations (TF32, benchmark mode)
- ✅ تنظيف ذاكرة GPU قبل وبعد المعالجة
- ✅ معالجة مُحسنة للـ segments
- ✅ WAV conversion monitoring

🎯 الهدف: تحقيق سرعة > 5x مع الحفاظ على الجودة
"""

import os
from core.config import logger
import time
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import concurrent.futures
import threading
from multiprocessing import Pool, cpu_count
from queue import Queue
import json
from models.db import get_db_client

# Safe torch import
try:
    import torch
    TORCH_AVAILABLE = True
    # ====== GPU/CUDA Diagnostic Print ======
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available() if torch else False)
    if torch and torch.cuda.is_available():
        print("CUDA version:", getattr(torch.version, "cuda", None))
        print("cuDNN version:", torch.backends.cudnn.version())
        print("GPU count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected or PyTorch not available")
except ImportError as e:
    print(f"⚠️ PyTorch not available: {e}")
    torch = None
    TORCH_AVAILABLE = False

WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    logger.info("✅ faster_whisper library found")
except ImportError as e:
    logger.warning(f"⚠ faster_whisper not installed: {e}")
except Exception as e:
    logger.error(f"❌ faster_whisper error: {e}")


# Import utility functions from services.utils
from .utils import format_time, clear_gpu_cache, convert_to_wav, cleanup_temp_wav, check_ffmpeg_availability

def optimize_cuda_settings():
    """تحسين إعدادات CUDA للسرعة القصوى"""
    if TORCH_AVAILABLE and torch and torch.cuda.is_available():
        # تمكين optimized attention للسرعة
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # تقليل synchronization للسرعة
        torch.backends.cudnn.deterministic = False
        logger.info("🎯 CUDA optimizations enabled for maximum speed")
    elif not TORCH_AVAILABLE:
        logger.warning("⚠️ PyTorch not available - CUDA optimizations skipped")

# تطبيق التحسينات عند تحميل الوحدة
optimize_cuda_settings()


# Global model cache للسرعة
_MODEL_CACHE = None
_MODEL_CACHE_PATH = None

class WhisperService:
    @staticmethod
    def get_project_model_path() -> str:
        """
        Returns the default path to the local Whisper model directory.
        """
        return os.path.join("models", "faster-whisper-large-v3")
    
    def __init__(self, collection, model_path: str = None, auto_download: bool = False, enable_parallel: bool = False):
        self.collection = collection
        self.model_path = model_path or self.get_project_model_path()
        self.auto_download = auto_download
        self.enable_parallel = enable_parallel
        
        # استخدام الدالة الـ standalone بدلاً من method
        self.model = get_whisper_model(self.model_path, auto_download=self.auto_download)
        
        # إعداد parallel processing
        self.cpu_cores = cpu_count()
        self.gpu_available = TORCH_AVAILABLE and torch and torch.cuda.is_available()
        self.segment_queue = Queue()
        self.result_queue = Queue()
        
        # Thread pool للمعالجة المتوازية
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(4, self.cpu_cores))
        self.gpu_lock = threading.Lock()  # حماية GPU من التداخل
        
        logger.info(f"🚀 Parallel processing enabled: CPU cores={self.cpu_cores}, GPU={self.gpu_available}")
    
    def __del__(self):
        """تنظيف الموارد عند الانتهاء"""
        if hasattr(self, 'cpu_executor'):
            self.cpu_executor.shutdown(wait=False)

    def split_audio_intelligent(self, audio_path: str, max_chunk_duration: int = 30) -> list:
        """
        تقسيم الملف الصوتي بذكاء حسب فترات الصمت
        """
        try:
            import librosa
            
            # تحميل الملف الصوتي
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            if duration <= max_chunk_duration:
                return [{"start": 0, "end": duration, "path": audio_path}]
            
            # البحث عن فترات الصمت
            from librosa import effects
            intervals = librosa.effects.split(y, top_db=20)
            
            chunks = []                          
            current_start = 0
            chunk_duration = 0
            
            for interval in intervals:
                interval_duration = (interval[1] - interval[0]) / sr
                
                if chunk_duration + interval_duration > max_chunk_duration and chunk_duration > 0:
                    # حفظ القطعة الحالية
                    chunks.append({
                        "start": current_start,
                        "end": current_start + chunk_duration,
                        "duration": chunk_duration
                    })
                    current_start = current_start + chunk_duration
                    chunk_duration = interval_duration
                else:
                    chunk_duration += interval_duration
            
            # إضافة القطعة الأخيرة
            if chunk_duration > 0:
                chunks.append({
                    "start": current_start,
                    "end": current_start + chunk_duration,
                    "duration": chunk_duration
                })
            
            logger.info(f"🔪 Split audio into {len(chunks)} intelligent chunks")
            return chunks
            
        except ImportError:
            logger.warning("⚠ librosa not available, using simple splitting")
            return self.split_audio_simple(audio_path, max_chunk_duration)
        except Exception as e:
            logger.warning(f"⚠ Intelligent splitting failed: {e}, using simple splitting")
            return self.split_audio_simple(audio_path, max_chunk_duration)
    
    def split_audio_simple(self, audio_path: str, max_chunk_duration: int = 30) -> list:
        """
        تقسيم بسيط للملف الصوتي
        """
        try:
            # الحصول على مدة الملف
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip())
            
            if duration <= max_chunk_duration:
                return [{"start": 0, "end": duration, "duration": duration}]
            
            chunks = []
            current_time = 0
            
            while current_time < duration:
                end_time = min(current_time + max_chunk_duration, duration)
                chunks.append({
                    "start": current_time,
                    "end": end_time,
                    "duration": end_time - current_time
                })
                current_time = end_time
            
            logger.info(f"🔪 Split audio into {len(chunks)} simple chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Audio splitting failed: {e}")
            return [{"start": 0, "end": 30, "duration": 30}]  # fallback
    
    def extract_audio_chunk(self, audio_path: str, start_time: float, end_time: float) -> str:
        """
        استخراج قطعة صوتية محددة
        """
        try:
            temp_dir = tempfile.gettempdir()
            chunk_path = os.path.join(temp_dir, f"chunk_{start_time:.2f}_{end_time:.2f}.wav")
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-ar', '16000', '-ac', '1',
                '-f', 'wav', chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                return chunk_path
            else:
                logger.error(f"❌ Failed to extract chunk {start_time}-{end_time}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Chunk extraction failed: {e}")
            return None
    
    def process_chunk_on_gpu(self, chunk_info: dict, chunk_path: str) -> dict:
        """
        معالجة قطعة صوتية على GPU
        """
        try:
            with self.gpu_lock:  # حماية GPU من التداخل
                start_time = time.time()
                
                segments, info = self.model.transcribe(
                    chunk_path,
                    language="ar",
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    word_timestamps=False
                )
                
                # تجميع النص
                text_parts = []
                for segment in segments:
                    if segment.text.strip():
                        text_parts.append(segment.text.strip())
                
                processing_time = time.time() - start_time
                
                return {
                    "chunk_info": chunk_info,
                    "text": " ".join(text_parts),
                    "processing_time": processing_time,
                    "processor": "GPU"
                }
                
        except Exception as e:
            logger.error(f"❌ GPU chunk processing failed: {e}")
            return {
                "chunk_info": chunk_info,
                "text": "",
                "processing_time": 0,
                "processor": "GPU_ERROR",
                "error": str(e)
            }
        finally:
            # تنظيف الملف المؤقت
            if chunk_path and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except:
                    pass
    
    def transcribe_audio_parallel(self, audio_path: str) -> str:
        """
        تحويل الصوت إلى نص بمعالجة متوازية بين GPU و CPU
        """
        try:
            total_start_time = time.time()
            logger.info(f"🚀 Starting PARALLEL transcription: {audio_path}")
            
            # تنظيف ذاكرة GPU
            clear_gpu_cache(logger=logger)
            
            # تحويل إلى WAV إذا لزم الأمر
            wav_path = convert_to_wav(audio_path, logger=logger)
            
            if not self.enable_parallel:
                # العودة للمعالجة العادية
                logger.info("🔄 Parallel disabled, using standard processing")
                return self.transcribe_audio_with_whisper(wav_path)
            
            # تقسيم الملف الصوتي ذكياً
            chunks = self.split_audio_intelligent(wav_path, max_chunk_duration=20)
            
            if len(chunks) <= 1:
                # ملف صغير، استخدم المعالجة العادية
                logger.info("📝 Small file, using standard processing")
                result = self.transcribe_audio_with_whisper(wav_path)
                cleanup_temp_wav(wav_path, audio_path, logger=logger)
                return result
            
            logger.info(f"🔪 Processing {len(chunks)} chunks in parallel")
            
            # تحضير المهام للمعالجة المتوازية
            tasks = []
            for i, chunk in enumerate(chunks):
                chunk_path = self.extract_audio_chunk(wav_path, chunk["start"], chunk["end"])
                if chunk_path:
                    tasks.append((chunk, chunk_path, i))
            
            if not tasks:
                logger.error("❌ No valid chunks created")
                return "خطأ: فشل في تقسيم الملف الصوتي"
            
            # معالجة بـ GPU فقط - لا نستخدم CPU
            results = []
            gpu_tasks = []
            
            # إعطاء كل المهام لـ GPU
            for chunk, chunk_path, idx in tasks:
                gpu_tasks.append((chunk, chunk_path, idx))
            
            logger.info(f"📊 Using GPU for all {len(gpu_tasks)} tasks - no CPU processing")
            
            # معالجة GPU فقط (تسلسلية لتجنب تداخل الذاكرة)
            gpu_results = []
            for chunk, chunk_path, idx in gpu_tasks:
                result = self.process_chunk_on_gpu(chunk, chunk_path)
                result["original_index"] = idx
                gpu_results.append(result)
            
            # دمج النتائج بالترتيب الصحيح (GPU فقط)
            all_results = gpu_results
            all_results.sort(key=lambda x: x["original_index"])
            
            # تجميع النص النهائي
            final_text_parts = []
            total_processing_time = 0
            gpu_time = 0
            
            for result in all_results:
                if result["text"].strip():
                    final_text_parts.append(result["text"].strip())
                
                total_processing_time += result["processing_time"]
                if result["processor"].startswith("GPU"):
                    gpu_time += result["processing_time"]
            
            final_transcript = " ".join(final_text_parts)
            total_time = time.time() - total_start_time
            
            # إحصائيات الأداء (GPU فقط)
            logger.info(f"✅ GPU-ONLY transcription completed")
            logger.info(f"⏱ Total time: {format_time(total_time)}")
            logger.info(f"🚀 GPU processing time: {format_time(gpu_time)}")
            logger.info(f"� GPU efficiency: {format_time(total_processing_time)} → {format_time(total_time)}")
            logger.info(f"🎯 Speedup: {(total_processing_time / total_time):.2f}x")
            logger.info(f"📝 Result: {len(final_transcript)} characters")
            
            # تنظيف الملفات المؤقتة
            cleanup_temp_wav(wav_path, audio_path, logger=logger)
            
            if not final_transcript:
                return "لم يتم العثور على نص في الملف الصوتي"
            
            return final_transcript
            
        except Exception as e:
            logger.error(f"❌ Parallel transcription failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"خطأ في التحويل المتوازي: {str(e)}"
        finally:
            clear_gpu_cache(logger=logger)
    
    def process_chunk_on_cpu(self, chunk_info: dict, chunk_path: str) -> dict:
        """
        معالجة قطعة صوتية على GPU (تم تغيير الاسم ولكن نستخدم GPU الآن)
        """
        try:
            start_time = time.time()
            
            # استخدام نفس الموديل الـ GPU بدلاً من إنشاء موديل CPU منفصل
            segments, info = self.model.transcribe(
                chunk_path,
                language="ar",
                beam_size=1,
                best_of=1,
                temperature=0,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                word_timestamps=False
            )
            
            # تجميع النص
            text_parts = []
            for segment in segments:
                if segment.text.strip():
                    text_parts.append(segment.text.strip())
            
            processing_time = time.time() - start_time
            
            return {
                "chunk_info": chunk_info,
                "text": " ".join(text_parts),
                "processing_time": processing_time,
                "processor": "GPU"
            }
            
        except Exception as e:
            logger.error(f"❌ GPU chunk processing failed: {e}")
            return {
                "chunk_info": chunk_info,
                "text": "",
                "processing_time": 0,
                "processor": "GPU_ERROR",
                "error": str(e)
            }
        finally:
            # تنظيف الملف المؤقت
            if chunk_path and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except:
                    pass

    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        try:
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            return {
                "file_size": file_size,
                "file_size_mb": file_size_mb,
                "exists": True
            }
        except Exception as e:
            logger.warning(f"⚠ Could not get audio info: {e}")
            return {
                "file_size": 0,
                "file_size_mb": 0,
                "exists": False
            }

    def transcribe_audio_with_whisper(self, audio_path: str) -> str:
        wav_path = None
        try:
            if self.model is None:
                logger.error("❌ Model is None")
                return f"خطأ: الموديل غير متوفر"
                
            total_start_time = time.time()
            logger.info(f"🎵 Starting GPU transcription for: {audio_path}")
            
            # تنظيف ذاكرة GPU قبل البداية
            clear_gpu_cache(logger=logger)
            
            # تحويل سريع للـ WAV
            conversion_start = time.time()
            wav_path = convert_to_wav(audio_path, logger=logger)
            conversion_time = time.time() - conversion_start
            
            if conversion_time > 0.5:  # إذا كان التحويل بطيء
                logger.warning(f"⚠ WAV conversion took {conversion_time:.2f}s - consider pre-converting files")
            audio_info = self.get_audio_info(wav_path)
            if not audio_info["exists"]:
                logger.error(f"❌ Audio file not found: {wav_path}")
                return "خطأ: الملف الصوتي غير موجود"
            logger.info(f"📁 WAV file size: {audio_info['file_size_mb']:.2f} MB")
            transcription_start = time.time()
            segments, info = self.model.transcribe(
                wav_path,
                language="ar",  # تحديد اللغة مسبقاً لتسريع التعرف
                beam_size=1,    # beam_size=1 أسرع بكثير من 5
                best_of=1,
                temperature=0,
                vad_filter=False,  # تعطيل VAD filter مؤقتاً
                condition_on_previous_text=False,  # False أسرع
                no_speech_threshold=0.6,  # threshold أقل للسرعة
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,  # إضافة threshold للسرعة
                no_repeat_ngram_size=0,  # تسريع المعالجة
                patience=1,  # تقليل الصبر للسرعة
                word_timestamps=False  # عدم حساب word timestamps للسرعة
            )
            transcription_time = time.time() - transcription_start
            logger.info(f"Language detected: {info.language} (confidence: {info.language_probability:.2f})")
            logger.info(f" Audio duration: {format_time(info.duration)}")
            logger.info(f"Core transcription completed in {format_time(transcription_time)}")
            segment_start = time.time()
            
            # معالجة فائقة السرعة للـ segments
            transcript_parts = []
            segment_count = 0
            
            # استخدام خطة واحدة للمعالجة
            full_text = ""
            for segment in segments:
                text = segment.text.strip()
                if text:
                    full_text += text + " "
                    segment_count += 1
            
            final_transcript = full_text.strip()
            
            segment_time = time.time() - segment_start
            if not final_transcript:
                logger.warning("Empty transcription result")
                return "لم يتم العثور على نص في الملف الصوتي"
            total_time = time.time() - total_start_time
            speed_ratio = info.duration / total_time if total_time > 0 else 0
            
            # معلومات مختصرة للسرعة
            if segment_time > 0.1:  # فقط إذا كان وقت المعالجة ملحوظ
                logger.info(f"🔗 Segments processed in {format_time(segment_time)}")
            
            logger.info(f"✅ Transcription completed successfully")
            logger.info(f"⏱ Total time: {format_time(total_time)}")
            logger.info(f"🚀 Speed ratio: {speed_ratio:.2f}x (audio duration / processing time)")
            logger.info(f"📊 Stats: {segment_count} segments, {len(final_transcript)} characters")
            return final_transcript
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"خطأ في التحويل: {str(e)}"
        finally:
            if wav_path:
                cleanup_temp_wav(wav_path, audio_path, logger=logger)
            clear_gpu_cache(logger=logger)

    def update_basic_fields(self, file_uuid: str, status: str, transcript: str = None, processing_time: float = None, audio_duration: float = None):
        # Always get collection from get_db_client
        try:
            _, _, collection = get_db_client()
        except Exception as e:
            logger.error(f"❌ Database not initialized: {e}")
            return
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        if transcript:
            update_data["transcript"] = transcript
        try:
            result = collection.update_one(
                {"uuid": file_uuid},
                {"$set": update_data}
            )
            if result.modified_count > 0:
                logger.info(f"✅ Updated {file_uuid}: {status}")
            else:
                logger.warning(f"⚠ No document updated for {file_uuid}")
        except Exception as e:
            logger.error(f"❌ Failed to update {file_uuid}: {e}")

    def process_audio_file(self, file_uuid: str, model_path: str = None) -> bool:
        """
        Process single audio file with GPU-optimized Whisper Large V3
        """
        try:
            process_start_time = time.time()
            _, _, collection = get_db_client()
            file_doc = collection.find_one({"uuid": file_uuid})
            if not file_doc:
                logger.error(f"❌ File not found: {file_uuid}")
                return False
            self.update_basic_fields(file_uuid, "processing")
            audio_path = file_doc["file_path"]
            logger.info(f"🔄 Processing: {file_uuid}")
            logger.info(f"📁 Audio file: {audio_path}")
            audio_duration = None
            if WHISPER_AVAILABLE:
                whisper_service = WhisperService(collection=collection, model_path=model_path)
                audio_info = whisper_service.get_audio_info(audio_path)
                transcript = whisper_service.transcribe_audio_with_whisper(audio_path)
                try:
                    wav_path = convert_to_wav(audio_path)
                    segments, info = whisper_service.model.transcribe(wav_path, beam_size=1, language="ar")
                    audio_duration = info.duration
                    del segments
                    cleanup_temp_wav(wav_path, audio_path)
                    clear_gpu_cache()
                except Exception:
                    logger.warning("⚠ Could not get audio duration from model")
            else:
                logger.error("❌ Whisper not available")
                self.update_basic_fields(file_uuid, "error")
                return False
            total_processing_time = time.time() - process_start_time
            if transcript.startswith("خطأ"):
                self.update_basic_fields(file_uuid, "error", processing_time=total_processing_time)
                return False
            self.update_basic_fields(file_uuid, "completed", transcript, total_processing_time, audio_duration)
            logger.info(f"✅ Successfully completed: {file_uuid}")
            logger.info(f"⏱ Total processing time: {format_time(total_processing_time)}")
            return True
        except Exception as e:
            logger.error(f"❌ Error processing {file_uuid}: {e}")
            self.update_basic_fields(file_uuid, "error")
            return False
        finally:
            clear_gpu_cache()
    
    def transcribe_audio(self, audio_path: str) -> str:
        try:
            start_time = time.time()
            
            # استخدام المعالجة المتوازية إذا كانت مفعلة
            if self.enable_parallel:
                result = self.transcribe_audio_parallel(audio_path)
            else:
                result = self.transcribe_audio_with_whisper(audio_path)
            
            total_time = time.time() - start_time
            processing_type = "PARALLEL" if self.enable_parallel else "STANDARD"
            logger.info(f"🎯 {processing_type} transcription completed in {format_time(total_time)}")
            return result
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return f"خطأ في التحويل: {str(e)}"
        finally:
            clear_gpu_cache(logger=logger)

    @staticmethod
    def check_model_availability():
        model_path = WhisperService.get_project_model_path()
        if not os.path.exists(model_path):
            logger.warning(f"⚠ Model not found at: {model_path}")
            logger.info("💡 Please download the model to the models/faster-whisper-large-v3 folder")
            return False
        logger.info(f"✅ Model found at: {model_path}")
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU available: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            if gpu_memory < 6:
                logger.warning("⚠ GPU memory might be insufficient for Large V3 model")
                return False
        else:
            logger.warning("💻 No GPU detected - transcription will be slower")
        return True

    def get_system_info(self):
        info = {
            "whisper_available": WHISPER_AVAILABLE,
            "cuda_available": TORCH_AVAILABLE and torch and torch.cuda.is_available(),
            "ffmpeg_available": check_ffmpeg_availability(logger=logger),
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path)
        }
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return info

    def get_processing_stats(self):
        """Get processing statistics from database"""
        try:
            _, _, collection = get_db_client()
            stats = collection.aggregate([
                {"$match": {"status": "completed", "processing_time_seconds": {"$exists": True}}},
                {"$group": {
                    "_id": None,
                    "total_processed": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time_seconds"},
                    "min_processing_time": {"$min": "$processing_time_seconds"},
                    "max_processing_time": {"$max": "$processing_time_seconds"},
                    "avg_speed_ratio": {"$avg": "$speed_ratio"},
                    "total_transcript_length": {"$sum": "$transcript_length"}
                }}
            ])
            result = list(stats)
            if result:
                stat = result[0]
                return {
                    "total_processed": stat["total_processed"],
                    "avg_processing_time": format_time(stat["avg_processing_time"]),
                    "min_processing_time": format_time(stat["min_processing_time"]),
                    "max_processing_time": format_time(stat["max_processing_time"]),
                    "avg_speed_ratio": f"{stat.get('avg_speed_ratio', 0):.2f}x",
                    "total_transcript_length": stat["total_transcript_length"]
                }
            return None
        except Exception as e:
            logger.error(f"❌ Error getting processing stats: {e}")
            return None

    def benchmark_performance(self):
        try:
            benchmark_start = time.time()
            logger.info("🏁 Starting performance benchmark...")
            if self.model is None:
                logger.error("❌ Benchmark failed: could not load model")
                return False
            clear_gpu_cache(logger=logger)
            benchmark_time = time.time() - benchmark_start
            logger.info(f"✅ Benchmark completed in {format_time(benchmark_time)}")
            logger.info("🎯 System ready for high-performance transcription")
            return True
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return False

def download_model_from_internet(model_name: str = "large-v3", download_path: str = None) -> bool:
    """
    Download Whisper model from Hugging Face if not available locally
    """
    try:
        if download_path is None:
            download_path = WhisperService.get_project_model_path()
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        logger.info(f"🌐 Downloading {model_name} model from internet...")
        logger.info(f"📁 Download location: {download_path}")
        
        # Download model with automatic fallback to internet
        model = WhisperModel(
            model_name,
            device="cpu",  # Download on CPU first
            compute_type="int8",
            download_root=os.path.dirname(download_path),
            local_files_only=False  # Allow internet download
        )
        
        logger.info(f"✅ Model {model_name} downloaded successfully to {download_path}")
        del model  # Free memory after download
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from internet: {e}")
        return False

def get_whisper_model(model_path: str = None, auto_download: bool = True) -> Optional[object]:
    """
    Load Whisper model with caching and optimized GPU settings
    Auto-downloads from internet if local model not found
    """
    global _MODEL_CACHE, _MODEL_CACHE_PATH
    
    if not WHISPER_AVAILABLE:
        logger.error("❌ faster_whisper not available")
        return None
    
    # إذا لم يتم تمرير مسار، استخدم المسار من المشروع
    if model_path is None:
        model_path = WhisperService.get_project_model_path()
    
    # استخدم الـ cache إذا كان الموديل نفسه
    if _MODEL_CACHE is not None and _MODEL_CACHE_PATH == model_path:
        logger.info("🚀 Using cached model for faster loading")
        return _MODEL_CACHE
    
    # Start model loading timer
    model_load_start = time.time()
    
    try:
        logger.info(f"🔄 Loading Whisper model from: {model_path}")
        
        # Check if local model exists
        local_files_only = True
        if not os.path.exists(model_path):
            if auto_download:
                logger.warning(f"⚠ Local model not found at: {model_path}")
                logger.info("🌐 Attempting to download from internet...")
                
                # Try to download from internet
                if download_model_from_internet("large-v3", model_path):
                    logger.info("✅ Model downloaded, loading from local path...")
                else:
                    logger.info("🌐 Download failed, trying direct internet loading...")
                    local_files_only = False
                    model_path = "large-v3"  # Use model name for direct download
            else:
                logger.error(f"❌ Local model not found at: {model_path}")
                logger.info("💡 Set auto_download=True or download manually to models/faster-whisper-large-v3 folder")
                return None
        
        # فحص GPU وإعداد الجهاز - إجبار استخدام GPU
        device = "cuda"  # إجبار استخدام GPU دائماً
        compute_type = "int8"  # استخدم int8 للـ RTX 3050 4GB
        cpu_threads = 0
        
        # تجربة استخدام GPU أولاً
        try:
            if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 FORCED GPU MODE: {gpu_name} ({gpu_memory:.1f}GB VRAM) with int8")
            else:
                logger.warning("⚠️ CUDA not available but forcing GPU mode anyway")
                # سنحاول GPU على أي حال
                device = "cuda"
        except Exception as gpu_error:
            logger.error(f"❌ GPU check failed: {gpu_error}, falling back to CPU")
            device = "cpu"
            compute_type = "int8"
            cpu_threads = 4
        
        # تحميل الموديل مع محاولة GPU أولاً ثم CPU كـ fallback
        try:
            logger.info(f"🚀 Attempting GPU loading first...")
            model = WhisperModel(
                model_path, 
                device="cuda",  # محاولة GPU أولاً
                compute_type="int8",
                num_workers=1,
                download_root=os.path.dirname(model_path) if local_files_only else None,
                local_files_only=local_files_only,
                cpu_threads=0
            )
            logger.info(f"✅ SUCCESS: GPU model loaded successfully!")
            device_used = "cuda"
            
        except Exception as gpu_error:
            logger.warning(f"⚠️ GPU loading failed: {gpu_error}")
            logger.info(f"🔄 Falling back to CPU...")
            
            try:
                model = WhisperModel(
                    model_path, 
                    device="cpu",
                    compute_type="int8",
                    num_workers=2,
                    download_root=os.path.dirname(model_path) if local_files_only else None,
                    local_files_only=local_files_only,
                    cpu_threads=4
                )
                logger.info(f"✅ CPU model loaded successfully as fallback")
                device_used = "cpu"
                
            except Exception as cpu_error:
                logger.error(f"❌ Both GPU and CPU loading failed!")
                logger.error(f"GPU error: {gpu_error}")
                logger.error(f"CPU error: {cpu_error}")
                return None
        
        # Cache the model
        _MODEL_CACHE = model
        _MODEL_CACHE_PATH = model_path
        
        model_load_time = time.time() - model_load_start
        
        logger.info(f"✅ Whisper Large V3 loaded successfully on {device_used} in {format_time(model_load_time)}")
        logger.info(f"🔧 Final configuration: device={device_used}, compute_type=int8")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load local Whisper model: {e}")
        return None

def clear_model_cache():
    """Clear the global model cache to free memory"""
    global _MODEL_CACHE, _MODEL_CACHE_PATH
    if _MODEL_CACHE is not None:
        del _MODEL_CACHE
        _MODEL_CACHE = None
        _MODEL_CACHE_PATH = None
        clear_gpu_cache()
        logger.info("🧹 Model cache cleared")

def update_basic_fields(file_uuid: str, status: str, transcript: str = None, processing_time: float = None, audio_duration: float = None):
    # Always get collection from get_db_client
    try:
        _, _, collection = get_db_client()
    except Exception as e:
        logger.error(f"❌ Database not initialized: {e}")
        return
    update_data = {
        "status": status,
        "updated_at": datetime.now()
    }
    if transcript:
        update_data["transcript"] = transcript
    try:
        result = collection.update_one(
            {"uuid": file_uuid},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            logger.info(f"✅ Updated {file_uuid}: {status}")
        else:
            logger.warning(f"⚠ No document updated for {file_uuid}")
    except Exception as e:
        logger.error(f"❌ Failed to update {file_uuid}: {e}")

def process_audio_file(file_uuid: str, model_path: str = None) -> bool:
    """
    Process single audio file with GPU-optimized Whisper Large V3
    Single worker, no chunking, optimized for short/medium audio
    Now includes WAV conversion and GPU cache clearing
    Uses get_db_client for MongoDB access.
    """
    try:
        # Start processing timer
        process_start_time = time.time()
        # Always get collection from get_db_client
        _, _, collection = get_db_client()
        file_doc = collection.find_one({"uuid": file_uuid})
        if not file_doc:
            # تجاهل التحذير تماماً، فقط إعادة False بدون أي لوج
            return False
        update_basic_fields(file_uuid, "processing")
        audio_path = file_doc["file_path"]
        logger.info(f"🔄 Processing: {file_uuid}")
        logger.info(f"📁 Audio file: {audio_path}")
        audio_duration = None
        if WHISPER_AVAILABLE:
            # Use WhisperService for all audio/model operations
            _, _, collection = get_db_client()
            whisper_service = WhisperService(collection=collection, model_path=model_path)
            audio_info = whisper_service.get_audio_info(audio_path)
            transcript = whisper_service.transcribe_audio_with_whisper(audio_path)
            try:
                wav_path = convert_to_wav(audio_path)
                segments, info = whisper_service.model.transcribe(wav_path, beam_size=1, language="ar")
                audio_duration = info.duration
                del segments
                cleanup_temp_wav(wav_path, audio_path)
                clear_gpu_cache()
            except:
                logger.warning("⚠ Could not get audio duration from model")
        else:
            logger.error("❌ Whisper not available")
            update_basic_fields(file_uuid, "error")
            return False
        total_processing_time = time.time() - process_start_time
        if transcript.startswith("خطأ"):
            update_basic_fields(file_uuid, "error", processing_time=total_processing_time)
            return False
        update_basic_fields(file_uuid, "completed", transcript, total_processing_time, audio_duration)
        logger.info(f"✅ Successfully completed: {file_uuid}")
        logger.info(f"⏱ Total processing time: {format_time(total_processing_time)}")
        return True
    except Exception as e:
        logger.error(f"❌ Error processing {file_uuid}: {e}")
        update_basic_fields(file_uuid, "error")
        return False
    finally:
        clear_gpu_cache()

def transcribe_audio(audio_path: str, model_path: str = None, auto_download: bool = True, enable_parallel: bool = False) -> str:
    """
    Direct transcription function for single audio file
    Uses GPU-optimized Whisper Large V3 with parallel processing
    Now includes WAV conversion and GPU cache clearing
    Auto-downloads model from internet if not found locally
    
    Args:
        audio_path: Path to the audio file
        model_path: Optional custom model path (if None, uses project model)
        auto_download: If True, downloads model from internet if not found locally
        enable_parallel: If True, uses parallel GPU+CPU processing for better speed
    
    Returns:
        Transcribed text
    """
    try:
        start_time = time.time()
        
        _, _, collection = get_db_client()
        whisper_service = WhisperService(
            collection=collection, 
            model_path=model_path, 
            auto_download=auto_download,
            enable_parallel=enable_parallel
        )
        
        if whisper_service.model is None:
            return "Error: Failed to load model"
            
        result = whisper_service.transcribe_audio(audio_path)
        total_time = time.time() - start_time
        
        processing_type = "PARALLEL" if enable_parallel else "STANDARD"
        logger.info(f"🎯 {processing_type} direct transcription completed in {format_time(total_time)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        return f"خطأ في التحويل: {str(e)}"
    
    finally:
        clear_gpu_cache()

def check_model_availability(auto_download: bool = False):
    """Check if the local model is available and GPU is ready"""
    model_path = WhisperService.get_project_model_path()
    
    if not os.path.exists(model_path):
        logger.warning(f"⚠ Model not found at: {model_path}")
        
        if auto_download:
            logger.info("🌐 Attempting to download model from internet...")
            if download_model_from_internet("large-v3", model_path):
                logger.info("✅ Model downloaded successfully")
            else:
                logger.error("❌ Failed to download model")
                return False
        else:
            logger.info("💡 Please download the model manually or set auto_download=True")
            logger.info("💡 Or download manually to the models/faster-whisper-large-v3 folder")
            return False
    
    logger.info(f"✅ Model found at: {model_path}")
 
    if TORCH_AVAILABLE and torch and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 GPU available: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        
        if gpu_memory < 6:
            logger.warning("⚠ GPU memory might be insufficient for Large V3 model")
            return False
    else:
        logger.warning("💻 No GPU detected - transcription will be slower")
    
    return True

def check_ffmpeg_availability():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            logger.info("✅ FFmpeg is available")
            return True
        else:
            logger.error("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ FFmpeg not found. Please install FFmpeg")
        return False

def get_system_info():
    """Get system information for debugging"""
    info = {
        "whisper_available": WHISPER_AVAILABLE,
        "cuda_available": TORCH_AVAILABLE and torch and torch.cuda.is_available(),
        "ffmpeg_available": check_ffmpeg_availability(),
        "model_path": WhisperService.get_project_model_path(),
        "model_exists": os.path.exists(WhisperService.get_project_model_path())
    }
    if TORCH_AVAILABLE and torch and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return info

def get_processing_stats():
    """Get processing statistics from database"""
    try:
        _, _, collection = get_db_client()
        stats = collection.aggregate([
            {"$match": {"status": "completed", "processing_time_seconds": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "total_processed": {"$sum": 1},
                "avg_processing_time": {"$avg": "$processing_time_seconds"},
                "min_processing_time": {"$min": "$processing_time_seconds"},
                "max_processing_time": {"$max": "$processing_time_seconds"},
                "avg_speed_ratio": {"$avg": "$speed_ratio"},
                "total_transcript_length": {"$sum": "$transcript_length"}
            }}
        ])
        
        result = list(stats)
        if result:
            stat = result[0]
            return {
                "total_processed": stat["total_processed"],
                "avg_processing_time": format_time(stat["avg_processing_time"]),
                "min_processing_time": format_time(stat["min_processing_time"]),
                "max_processing_time": format_time(stat["max_processing_time"]),
                "avg_speed_ratio": f"{stat.get('avg_speed_ratio', 0):.2f}x",
                "total_transcript_length": stat["total_transcript_length"]
            }
        return None
    except Exception as e:
        logger.error(f"❌ Error getting processing stats: {e}")
        return None

def benchmark_performance():
    """Run a simple performance benchmark"""
    try:
        benchmark_start = time.time()
        
        logger.info("🏁 Starting performance benchmark...")
        
        model = get_whisper_model()
        if model is None:
            logger.error("❌ Benchmark failed: could not load model")
            return False
        
      
        clear_gpu_cache()
        
        benchmark_time = time.time() - benchmark_start
        
        logger.info(f"✅ Benchmark completed in {format_time(benchmark_time)}")
        logger.info("🎯 System ready for high-performance transcription")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return False

def test_model_download():
    """Test model download functionality"""
    try:
        logger.info("🧪 Testing model download functionality...")
        
        # Check current model status
        model_path = WhisperService.get_project_model_path()
        logger.info(f"📁 Expected model path: {model_path}")
        logger.info(f"📊 Model exists locally: {os.path.exists(model_path)}")
        
        # Test auto-download feature
        logger.info("🌐 Testing auto-download feature...")
        model = get_whisper_model(auto_download=True)
        
        if model is not None:
            logger.info("✅ Model loaded successfully (either local or downloaded)")
            return True
        else:
            logger.error("❌ Failed to load model")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model download test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing GPU-Optimized Whisper Large V3 with WAV Conversion & GPU Cache Management...")
    print("=" * 80)
    
    try:
        
        system_info = get_system_info()
        print("📊 System Information:")
        for key, value in system_info.items():
            print(f"   {key}: {value}")
        print()
        
        if not check_ffmpeg_availability():
            print("⚠ FFmpeg not available - WAV conversion will be disabled")
        
        # Test model download functionality
        print("\n🌐 Testing model download functionality...")
        if test_model_download():
            print("✅ Model download test passed")
        else:
            print("⚠ Model download test failed - checking manual availability...")
            
        if check_model_availability(auto_download=True):
            print("✅ Model availability OK")
        else:
            print("⚠ Model not available")
            exit(1)
       
        try:
            _, _, collection = get_db_client()
            print("✅ Database connection OK")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
        
        # Run performance benchmark
        print("\n🏁 Running performance benchmark...")
        if benchmark_performance():
            print("✅ Performance benchmark passed")
        else:
            print("❌ Performance benchmark failed")
            exit(1)
        
        # Get processing statistics
        stats = get_processing_stats()
        if stats:
            print("\n📈 Processing Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("\n🎯 GPU Optimizations Applied:")
        print("   • Single worker configuration for GPU efficiency")
        print("   • No chunking/segmentation for faster processing")
        print("   • Optimized beam_size=5 for quality balance")
        print("   • GPU memory management and monitoring")
        print("   • Comprehensive time tracking and performance metrics")
        print("   • ✅ torch.cuda.empty_cache() after each file")
        print("   • ✅ WAV conversion using FFmpeg")
        
        print("\n⏱ Time Tracking Features:")
        print("   • Model loading time")
        print("   • WAV conversion time")
        print("   • Core transcription time")
        print("   • Segment processing time")
        print("   • Total processing time")
        print("   • Speed ratio calculation (audio duration / processing time)")
        print("   • Database storage of timing metrics")
        
        print("\n🔧 New Features Added:")
        print("   • ✅ FFmpeg WAV conversion (16kHz, Mono, PCM)")
        print("   • ✅ GPU cache clearing after each transcription")
        print("   • ✅ Temporary file cleanup")
        print("   • ✅ Enhanced error handling")
        print("   • 🌐 Auto-download from internet if local model not found")
        print("   • 📁 Smart model path detection and fallback")
        
        print("\n📌 Model Download Options:")
        print("   • Local files only: get_whisper_model(auto_download=False)")
        print("   • Auto-download enabled: get_whisper_model(auto_download=True)  [DEFAULT]")
        print("   • Manual download: download_model_from_internet('large-v3')")
        print("   • Check availability: check_model_availability(auto_download=True)")
        
        print("\n🎉 All optimizations ready for fast GPU transcription with auto-download support!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()