import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@patch('src.api.app.AIGrowthEngineCore')
@patch('src.api.app.TaskScheduler')
def test_api_health_check(mock_scheduler, mock_engine):
    """测试API健康检查"""
    from src.api.app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch('src.api.app.get_engine')
def test_get_daily_report(mock_get_engine):
    """测试获取每日报告"""
    from src.api.app import app

    # 模拟引擎
    mock_engine = Mock()
    mock_engine.state = {
        'current_analysis': {
            'data_summary': {'test': 'data'},
            'insights': {'daily_report': 'Test report'},
            'anomalies': {},
            'recommendations': []
        }
    }
    mock_get_engine.return_value = mock_engine

    client = TestClient(app)
    response = client.get("/api/v1/reports/daily")

    assert response.status_code == 200
    data = response.json()
    assert 'date' in data
    assert 'metrics' in data
    assert 'insights' in data