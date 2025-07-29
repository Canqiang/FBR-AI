from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

# 请求模型
class PredictionRequest(BaseModel):
    """预测请求"""
    prediction_type: str = Field(..., description="预测类型：sales/demand/customer")
    periods: int = Field(7, ge=1, le=30, description="预测周期数")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99)
    historical_days: int = Field(90, ge=30, le=365)

class InsightRequest(BaseModel):
    """洞察请求"""
    query: str = Field(..., description="用户问题")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")

# 响应模型
class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    engine_status: str

class MetricSummary(BaseModel):
    """指标摘要"""
    revenue: Decimal
    orders: int
    customers: int
    avg_order_value: Decimal
    change_percentage: float

class AnomalySummary(BaseModel):
    """异常摘要"""
    type: str
    severity: str
    description: str
    detected_at: datetime
    affected_metrics: List[str]

class RecommendationResponse(BaseModel):
    """建议响应"""
    type: str
    priority: str
    action: str
    reason: str
    expected_impact: str
    confidence: float = Field(..., ge=0, le=1)

class DailyReportResponse(BaseModel):
    """每日报告响应"""
    date: datetime
    metrics: Dict[str, Any]
    insights: str
    anomalies: List[AnomalySummary]
    recommendations: List[RecommendationResponse]

class PredictionResponse(BaseModel):
    """预测响应"""
    prediction_type: str
    periods: int
    predictions: List[Dict[str, Any]]
    confidence_intervals: Optional[Dict[str, List[float]]]
    model_performance: Optional[Dict[str, float]]

class InsightResponse(BaseModel):
    """洞察响应"""
    query: str
    insight: str
    data_sources: List[str]
    confidence: float
    timestamp: datetime

class CustomerSegment(BaseModel):
    """客户分群"""
    segment_name: str
    customer_count: int
    total_revenue: Decimal
    avg_order_value: Decimal
    characteristics: Dict[str, Any]

class CustomerSegmentResponse(BaseModel):
    """客户分群响应"""
    segments: List[CustomerSegment]
    total_customers: int
    analysis_date: datetime

class ItemPerformance(BaseModel):
    """商品表现"""
    item_id: str
    item_name: str
    category: str
    revenue: Decimal
    units_sold: int
    avg_price: Decimal
    unique_buyers: int
    performance_score: float

class ItemPerformanceResponse(BaseModel):
    """商品表现响应"""
    items: List[ItemPerformance]
    period_days: int
    total_items: int

class AnalysisTaskResponse(BaseModel):
    """分析任务响应"""
    task_id: str
    status: str
    message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

# 错误响应
class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)