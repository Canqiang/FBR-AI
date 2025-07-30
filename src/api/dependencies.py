"""API依赖项"""
import logging
from fastapi import HTTPException
from typing import Optional

logger = logging.getLogger(__name__)

# 全局引擎实例（从app.py中引用）
_engine = None


def set_engine(engine):
    """设置引擎实例（由app.py在启动时调用）"""
    global _engine
    _engine = engine


def get_engine():
    """获取引擎实例的依赖函数"""
    if not _engine:
        logger.error("Engine not initialized")
        raise HTTPException(status_code=503, detail="Engine not available")
    return _engine