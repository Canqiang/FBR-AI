from fastapi import HTTPException
from ..engine.core import AIGrowthEngineCore

# 这个engine对象要和app.py一致，如果是全局单例
engine = None

def set_engine(global_engine):
    global engine
    engine = global_engine

def get_engine() -> AIGrowthEngineCore:
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine
