# src/api/app.py
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

logger = logging.getLogger(__name__)

# 全局引擎实例
engine = None
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine, scheduler

    logger.info("Starting AI Growth Engine API")

    try:
        # 初始化配置
        from config.settings import get_settings
        settings = get_settings()

        # 初始化引擎
        from src.engine.core import AIGrowthEngineCore
        engine = AIGrowthEngineCore()
        logger.info("Engine initialized successfully")

        # 初始化调度器（可选）
        try:
            from src.engine.scheduler import TaskScheduler
            scheduler = TaskScheduler(engine)
            scheduler.add_daily_analysis("08:00")
            scheduler.add_hourly_monitoring()
            scheduler.start()
            logger.info("Scheduler started successfully")
        except Exception as e:
            logger.warning(f"Scheduler initialization failed: {e}")
            scheduler = None

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        engine = None

    yield

    # 清理资源
    logger.info("Shutting down AI Growth Engine API")
    if scheduler:
        try:
            scheduler.stop()
        except:
            pass


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

# 导入路由
from src.api.routes import router

# 添加路由
app.include_router(router, prefix="/api/v1")


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "FBR AI Growth Engine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# 健康检查 - 在根路径下
@app.get("/health")
async def health_check():
    """健康检查端点"""
    global engine

    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "engine_status": "running" if engine else "not initialized",
        "scheduler_status": "running" if scheduler and scheduler.running else "stopped"
    }

    # 检查各个组件
    if engine:
        try:
            # 简单测试
            health_status["data_connection"] = "ok" if hasattr(engine, 'order_repo') else "error"
        except:
            health_status["data_connection"] = "error"
    else:
        health_status["data_connection"] = "not initialized"

    # 确定整体健康状态
    if engine is None:
        health_status["status"] = "degraded"
        health_status["message"] = "Engine not initialized"

    return health_status


# 获取引擎实例的依赖
def get_engine():
    """获取引擎实例"""
    if not engine:
        logger.error("Engine not initialized")
        # 尝试创建一个新实例
        try:
            from src.engine.core import AIGrowthEngineCore
            return AIGrowthEngineCore()
        except Exception as e:
            logger.error(f"Failed to create engine instance: {e}")
            raise HTTPException(status_code=503, detail="Engine not available")
    return engine


# 开发模式下的自动重载
if __name__ == "__main__":
    import os

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 获取端口
    port = int(os.getenv("API_PORT", 8000))

    # 启动服务器
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )