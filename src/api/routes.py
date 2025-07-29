from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, Depends
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from .schemas import (
    DailyReportResponse,
    PredictionRequest,
    PredictionResponse,
    InsightRequest,
    InsightResponse,
    RecommendationResponse,
    CustomerSegmentResponse,
    ItemPerformanceResponse
)
from ..engine.core import AIGrowthEngineCore
# from .app import get_engine
from .dependencies import get_engine

router = APIRouter()


@router.get("/reports/daily", response_model=DailyReportResponse)
async def get_daily_report(
        date: Optional[datetime] = Query(None, description="报告日期，默认为今天"),
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """获取每日报告"""
    if not date:
        date = datetime.now().date()

    # 获取或生成报告
    if engine.state.get('last_run') and engine.state['last_run'].date() == date:
        # 使用缓存的结果
        analysis = engine.state['current_analysis']
    else:
        # 运行新的分析
        analysis = engine.run_daily_analysis()

    return {
        "date": date,
        "metrics": analysis.get('data_summary', {}),
        "insights": analysis.get('insights', {}).get('daily_report', ''),
        "anomalies": analysis.get('anomalies', {}),
        "recommendations": analysis.get('recommendations', [])
    }


@router.post("/predictions", response_model=PredictionResponse)
async def create_prediction(
        request: PredictionRequest,
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """创建预测"""
    try:
        # 获取历史数据
        data = engine.analytics_repo.get_time_series_data(days=request.historical_days)

        # 运行预测
        if request.prediction_type == "sales":
            forecast = engine.predictor.predict_with_prophet(
                data,
                periods=request.periods
            )

            return {
                "prediction_type": request.prediction_type,
                "periods": request.periods,
                "predictions": forecast.to_dict('records'),
                "confidence_intervals": {
                    "lower": forecast['yhat_lower'].tolist(),
                    "upper": forecast['yhat_upper'].tolist()
                }
            }

        elif request.prediction_type == "demand":
            item_data = engine.order_repo.get_item_performance(days=request.historical_days)
            demand_prediction = engine.predictor.predict_item_demand(item_data)

            return {
                "prediction_type": request.prediction_type,
                "periods": request.periods,
                "predictions": demand_prediction['predictions'].tolist(),
                "feature_importance": demand_prediction['feature_importance'].to_dict('records')
            }

        else:
            raise HTTPException(status_code=400, detail="Unsupported prediction type")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights", response_model=InsightResponse)
async def generate_insights(
        request: InsightRequest,
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """生成AI洞察"""
    try:
        # 使用AI代理生成洞察
        response = engine.ai_agent.chat(request.query)

        return {
            "query": request.query,
            "insight": response['answer'],
            "data_sources": [step[0].tool for step in response.get('intermediate_steps', [])],
            "confidence": 0.85,  # 示例置信度
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
        category: Optional[str] = Query(None, description="建议类别：pricing/inventory/marketing"),
        limit: int = Query(10, ge=1, le=50),
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """获取智能建议"""
    try:
        # 获取最新的建议
        if engine.state.get('current_analysis'):
            all_recommendations = engine.state['current_analysis'].get('recommendations', [])
        else:
            # 运行快速分析生成建议
            analysis = engine.run_daily_analysis()
            all_recommendations = analysis.get('recommendations', [])

        # 过滤类别
        if category:
            filtered = [r for r in all_recommendations if r.get('type') == category]
        else:
            filtered = all_recommendations

        # 限制数量
        return filtered[:limit]

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers/segments", response_model=CustomerSegmentResponse)
async def get_customer_segments(engine: AIGrowthEngineCore = Depends(get_engine)):
    """获取客户分群分析"""
    try:
        segments_df = engine.customer_repo.get_customer_segments()

        segments = []
        for _, row in segments_df.iterrows():
            segments.append({
                "segment_name": row['segment'],
                "customer_count": int(row['customer_count']),
                "total_revenue": float(row['total_revenue']),
                "avg_order_value": float(row['avg_order_value']),
                "characteristics": {
                    "avg_order_count": float(row['avg_order_count'])
                }
            })

        return {
            "segments": segments,
            "total_customers": sum(s['customer_count'] for s in segments),
            "analysis_date": datetime.now()
        }

    except Exception as e:
        logger.error(f"Failed to get customer segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items/performance", response_model=ItemPerformanceResponse)
async def get_item_performance(
        days: int = Query(30, ge=1, le=365),
        category: Optional[str] = None,
        limit: int = Query(20, ge=1, le=100),
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """获取商品销售表现"""
    try:
        items_df = engine.order_repo.get_item_performance(days=days, top_n=limit)

        # 过滤类别
        if category:
            items_df = items_df[items_df['category_name'] == category]

        items = []
        for _, row in items_df.iterrows():
            items.append({
                "item_id": row.get('item_id', ''),
                "item_name": row['item_name'],
                "category": row['category_name'],
                "revenue": float(row['revenue']),
                "units_sold": int(row['units_sold']),
                "avg_price": float(row['avg_price']),
                "unique_buyers": int(row['unique_buyers']),
                "performance_score": float(row['revenue']) / items_df['revenue'].max()
            })

        return {
            "items": items,
            "period_days": days,
            "total_items": len(items)
        }

    except Exception as e:
        logger.error(f"Failed to get item performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/run")
async def run_analysis(
        background_tasks: BackgroundTasks,
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """触发运行分析（异步）"""
    task_id = f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 添加后台任务
    background_tasks.add_task(engine.run_daily_analysis)

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Analysis task has been queued"
    }


@router.get("/analysis/status/{task_id}")
async def get_analysis_status(
        task_id: str,
        engine: AIGrowthEngineCore = Depends(get_engine)
):
    """获取分析任务状态"""
    # 简化的状态检查
    if engine.state.get('last_run'):
        return {
            "task_id": task_id,
            "status": "completed",
            "completed_at": engine.state['last_run'],
            "results_available": True
        }
    else:
        return {
            "task_id": task_id,
            "status": "pending",
            "results_available": False
        }
