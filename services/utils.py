import os
import time
import subprocess
import tempfile
from pathlib import Path
import logging

# Safe torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

def clear_gpu_cache(logger=None):
    if TORCH_AVAILABLE and torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger:
            logger.info("🧹 GPU cache cleared")
    elif logger:
        logger.info("⚠️ PyTorch/CUDA not available - GPU cache clearing skipped")

def convert_to_wav(input_path: str, output_path: str = None, logger=None) -> str:
    try:
        conversion_start = time.time()
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            filename = Path(input_path).stem
            output_path = os.path.join(temp_dir, f"{filename}_converted.wav")
        if logger:
            logger.info(f"🎵 Converting to WAV: {input_path} -> {output_path}")
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-y',
            output_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        conversion_time = time.time() - conversion_start
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            if logger:
                logger.info(f"✅ WAV conversion completed in {format_time(conversion_time)}")
                logger.info(f"📁 WAV file size: {file_size:.2f} MB")
            return output_path
        else:
            if logger:
                logger.error("❌ WAV file not created")
            return input_path
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"❌ FFmpeg conversion failed: {e}")
            logger.error(f"stderr: {e.stderr}")
        return input_path
    except Exception as e:
        if logger:
            logger.error(f"❌ Conversion error: {e}")
        return input_path

def cleanup_temp_wav(wav_path: str, original_path: str, logger=None):
    try:
        if wav_path != original_path and os.path.exists(wav_path):
            os.remove(wav_path)
            if logger:
                logger.info(f"🗑 Cleaned up temporary WAV: {wav_path}")
    except Exception as e:
        if logger:
            logger.warning(f"⚠ Could not clean up temp file: {e}")

def check_ffmpeg_availability(logger=None):
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            if logger:
                logger.info("✅ FFmpeg is available")
            return True
        else:
            if logger:
                logger.error("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        if logger:
            logger.error("❌ FFmpeg not found. Please install FFmpeg")
        return False
