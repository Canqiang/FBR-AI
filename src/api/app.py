import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from config.settings import get_settings
from ..engine.core import AIGrowthEngineCore
from ..engine.scheduler import TaskScheduler
from .routes import router
from .schemas import HealthResponse

logger = logging.getLogger(__name__)

# 全局引擎实例
engine = None
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine, scheduler

    logger.info("Starting AI Growth Engine API")

    # 初始化引擎
    engine = AIGrowthEngineCore()

    # 初始化调度器
    scheduler = TaskScheduler(engine)
    scheduler.add_daily_analysis("08:00")
    scheduler.add_hourly_monitoring()
    scheduler.start()

    yield

    # 清理资源
    logger.info("Shutting down AI Growth Engine API")
    if scheduler:
        scheduler.stop()


# 创建FastAPI应用
app = FastAPI(
    title="FBR AI Growth Engine API",
    description="智能餐饮增长引擎API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加路由
app.include_router(router, prefix="/api/v1")


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# 健康检查
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "engine_status": "running" if engine else "not initialized"
    }


# 获取引擎实例的依赖
def get_engine() -> AIGrowthEngineCore:
    """获取引擎实例"""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine