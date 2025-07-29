from datetime import datetime

import pytest
from unittest.mock import Mock, patch
from src.engine.core import AIGrowthEngineCore
from src.engine.scheduler import TaskScheduler
import pandas as pd

class TestAIGrowthEngineCore:
    """测试核心引擎"""

    @patch('src.engine.core.OrderRepository')
    @patch('src.engine.core.CustomerRepository')
    @patch('src.engine.core.AnalyticsRepository')
    def test_engine_initialization(self, mock_analytics, mock_customer, mock_order):
        """测试引擎初始化"""
        engine = AIGrowthEngineCore()

        # 验证组件初始化
        assert engine.order_repo is not None
        assert engine.customer_repo is not None
        assert engine.analytics_repo is not None
        assert engine.predictor is not None
        assert engine.causal_engine is not None
        assert engine.anomaly_detector is not None
        assert engine.optimizer is not None
        assert engine.ai_agent is not None
        assert engine.analysis_chains is not None

    @patch('src.engine.core.OrderRepository')
    @patch('src.engine.core.CustomerRepository')
    @patch('src.engine.core.AnalyticsRepository')
    def test_run_daily_analysis(self, mock_analytics, mock_customer, mock_order):
        """测试每日分析运行"""
        # 设置模拟返回值
        mock_order.return_value.get_daily_sales.return_value = pd.DataFrame()
        mock_customer.return_value.get_customer_segments.return_value = pd.DataFrame()
        mock_analytics.return_value.get_time_series_data.return_value = pd.DataFrame({
            'ds': pd.date_range(end=datetime.now(), periods=30),
            'y': [50000] * 30
        })

        engine = AIGrowthEngineCore()

        # 模拟分析组件
        engine.anomaly_detector = Mock()
        engine.anomaly_detector.detect_sales_anomalies.return_value = pd.DataFrame()

        engine.predictor = Mock()
        engine.predictor.predict_with_prophet.return_value = pd.DataFrame()

        # 运行分析
        result = engine.run_daily_analysis()

        # 验证结果
        assert result['status'] == 'success'
        assert 'timestamp' in result
        assert 'duration' in result
        assert 'data_summary' in result


class TestTaskScheduler:
    """测试任务调度器"""

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        mock_engine = Mock()
        scheduler = TaskScheduler(mock_engine)

        assert scheduler.engine == mock_engine
        assert scheduler.jobs == {}
        assert scheduler.running == False

    def test_add_daily_analysis_task(self):
        """测试添加每日分析任务"""
        mock_engine = Mock()
        scheduler = TaskScheduler(mock_engine)

        scheduler.add_daily_analysis("09:00")

        assert 'daily_analysis' in scheduler.jobs
        assert scheduler.jobs['daily_analysis'] is not None

    @patch('src.engine.scheduler.schedule')
    def test_scheduler_start_stop(self, mock_schedule):
        """测试调度器启动和停止"""
        mock_engine = Mock()
        scheduler = TaskScheduler(mock_engine)

        # 测试启动
        scheduler.start()
        assert scheduler.running == True
        assert scheduler.thread is not None

        # 测试停止
        scheduler.stop()
        assert scheduler.running == False