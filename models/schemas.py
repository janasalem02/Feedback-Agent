from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# =================== Basic Response Models ===================

class UploadResponse(BaseModel):
	success: bool
	message: str
	file_uuid: Optional[str] = None

class HealthResponse(BaseModel):
	status: str
	database: Optional[str] = None
	ollama: Optional[str] = None
	whisper: Optional[str] = None
	simple_analysis: Optional[str] = None
	error: Optional[str] = None

# =================== Analysis Models ===================

class AnalysisRequest(BaseModel):
	text: str

class AnalysisResponse(BaseModel):
	success: bool
	classification: Optional[str] = None
	summary: Optional[str] = None
	error: Optional[str] = None

class SimpleAnalysisRequest(BaseModel):
	"""طلب التحليل البسيط"""
	text: str
	client_id: Optional[str] = None

class SimpleAnalysisData(BaseModel):
	"""نموذج بيانات التحليل البسيط"""
	basic_positive_points: List[str] = []
	basic_negative_points: List[str] = []
	linked_positive_points: List[str] = []
	linked_negative_points: List[str] = []
	total_points: int = 0
	criteria_matches: int = 0
	confidence: float = 0.0
	processing_time: float = 0.0
	client_id: str = ""

class SimpleAnalysisResponse(BaseModel):
	"""استجابة التحليل البسيط"""
	success: bool
	basic_positive_points: List[str] = []
	basic_negative_points: List[str] = []
	linked_positive_points: List[str] = []
	linked_negative_points: List[str] = []
	total_points: int = 0
	criteria_matches: int = 0
	confidence: float = 0.0
	processing_time: float = 0.0
	client_id: str = ""
	error: Optional[str] = None

# =================== Complete Analysis Models ===================

class MappedPoint(BaseModel):
	"""نقطة مربوطة بمعيار"""
	point: str
	criteria_id: Optional[str] = None
	criteria_name: Optional[str] = None
	weight: float = 0.0

class ClassificationData(BaseModel):
	"""بيانات التصنيف مع السكورات"""
	sentiment: str  # "Positive", "Negative", "Mixed"
	positive_score: float  # النسبة المئوية للإيجابي
	negative_score: float  # النسبة المئوية للسلبي

class ScoresData(BaseModel):
	"""تفاصيل حساب السكورات"""
	total_positive_weight: float
	total_negative_weight: float
	total_weight: float
	positive_percentage: float
	negative_percentage: float
	score_difference: float

class AnalysisMetadata(BaseModel):
	"""معلومات إضافية عن التحليل"""
	total_positive_points: int
	total_negative_points: int
	total_points: int
	matched_positive_points: int
	matched_negative_points: int
	analysis_timestamp: datetime
	llama_model_used: str
	client_id: str

class CompleteAnalysisData(BaseModel):
	"""التحليل الكامل مع كل التفاصيل - مطابق لـ MongoDB schema"""
	positive_points: List[MappedPoint] = []
	negative_points: List[MappedPoint] = []
	classification: ClassificationData
	scores: ScoresData
	metadata: AnalysisMetadata

# =================== File Models ===================

class FileInfo(BaseModel):
	"""معلومات الملف الكاملة"""
	uuid: str
	filename: str
	status: str
	created_at: datetime
	updated_at: Optional[datetime] = None
	file_path: Optional[str] = None
	file_size: Optional[int] = None
	transcript: Optional[str] = None
	classification: Optional[str] = None
	summary: Optional[str] = None
	processing_time_seconds: Optional[float] = None
    
	# التحليل البسيط والكامل
	simple_analysis: Optional[SimpleAnalysisData] = None
	analysis: Optional[CompleteAnalysisData] = None
    
	# معلومات إضافية
	error_message: Optional[str] = None
	client_id: Optional[str] = None

class FileListResponse(BaseModel):
	"""قائمة الملفات"""
	success: bool
	files: List[FileInfo] = []
	total_count: int = 0
	error: Optional[str] = None

class FileDetailResponse(BaseModel):
	"""تفاصيل ملف واحد"""
	success: bool
	file: Optional[FileInfo] = None
	error: Optional[str] = None

# =================== Complete Processing Models ===================

class CompleteAnalysisRequest(BaseModel):
	"""طلب التحليل الكامل للملف الصوتي"""
	audio_file_path: str
	client_id: str
	file_uuid: Optional[str] = None

class CompleteAnalysisResponse(BaseModel):
	"""استجابة التحليل الكامل"""
	success: bool
	uuid: Optional[str] = None
	transcript: Optional[str] = None
	classification: Optional[str] = None
	summary: Optional[str] = None
	analysis: Optional[CompleteAnalysisData] = None
	processing_time: Optional[float] = None
	steps_timing: Optional[Dict[str, float]] = None
	error: Optional[str] = None

# =================== Criteria Models ===================

class CriteriaItem(BaseModel):
	"""معيار واحد"""
	id: str
	name: str
	weight: float = Field(..., ge=0.0, le=1.0)  # وزن بين 0 و 1

class ClientCriteria(BaseModel):
	"""معايير العميل"""
	client_id: str = Field(..., alias="_id")
	criteria: List[CriteriaItem]
    
	class Config:
		allow_population_by_field_name = True

class CriteriaResponse(BaseModel):
	"""استجابة معايير العميل"""
	success: bool
	client_id: Optional[str] = None
	criteria: Optional[List[CriteriaItem]] = None
	total_criteria: Optional[int] = None
	error: Optional[str] = None

# =================== Statistics Models ===================

class ProcessingStats(BaseModel):
	"""إحصائيات المعالجة"""
	total_processed: int
	avg_processing_time: str
	min_processing_time: str
	max_processing_time: str
	avg_speed_ratio: str
	total_transcript_length: int

class SystemInfo(BaseModel):
	"""معلومات النظام"""
	whisper_available: bool
	cuda_available: bool
	ffmpeg_available: bool
	model_path: str
	model_exists: bool
	gpu_name: Optional[str] = None
	gpu_memory_gb: Optional[float] = None

class StatsResponse(BaseModel):
	"""استجابة الإحصائيات"""
	success: bool
	processing_stats: Optional[ProcessingStats] = None
	system_info: Optional[SystemInfo] = None
	error: Optional[str] = None

# =================== Error Models ===================

class ErrorDetail(BaseModel):
	"""تفاصيل الخطأ"""
	error_type: str
	error_message: str
	timestamp: datetime
	file_uuid: Optional[str] = None
	step: Optional[str] = None

class ErrorResponse(BaseModel):
	"""استجابة الخطأ"""
	success: bool = False
	error: str
	details: Optional[ErrorDetail] = None
	suggestions: Optional[List[str]] = None

# =================== Pagination Models ===================

class PaginationParams(BaseModel):
	"""معاملات التصفح"""
	page: int = Field(1, ge=1)
	limit: int = Field(10, ge=1, le=100)
	sort_by: Optional[str] = "created_at"
	sort_order: Optional[str] = Field("desc", regex="^(asc|desc)$")

class PaginatedResponse(BaseModel):
	"""استجابة مع تصفح"""
	success: bool
	data: List[Any] = []
	pagination: Dict[str, Any] = {}
	error: Optional[str] = None

# =================== Search Models ===================

class SearchParams(BaseModel):
	"""معاملات البحث"""
	query: Optional[str] = None
	status: Optional[str] = None
	client_id: Optional[str] = None
	classification: Optional[str] = None
	date_from: Optional[datetime] = None
	date_to: Optional[datetime] = None

class SearchRequest(BaseModel):
	"""طلب البحث"""
	search: SearchParams
	pagination: Optional[PaginationParams] = None

class SearchResponse(BaseModel):
	"""استجابة البحث"""
	success: bool
	results: List[FileInfo] = []
	total_found: int = 0
	search_time: float = 0.0
	pagination: Optional[Dict[str, Any]] = None
	error: Optional[str] = None

# =================== Export Models ===================

class ExportRequest(BaseModel):
	"""طلب تصدير البيانات"""
	format: str = Field(..., regex="^(json|csv|excel)$")
	filters: Optional[SearchParams] = None
	include_analysis: bool = True
	include_transcript: bool = True

class ExportResponse(BaseModel):
	"""استجابة التصدير"""
	success: bool
	download_url: Optional[str] = None
	file_size: Optional[int] = None
	records_count: Optional[int] = None
	expires_at: Optional[datetime] = None
	error: Optional[str] = None

# =================== Validation Models ===================

class ValidationError(BaseModel):
	"""خطأ التحقق"""
	field: str
	message: str
	value: Any

class ValidationResponse(BaseModel):
	"""استجابة التحقق"""
	valid: bool
	errors: List[ValidationError] = []
	warnings: List[str] = [] 
