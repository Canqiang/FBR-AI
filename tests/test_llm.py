from datetime import datetime

import pytest
from unittest.mock import Mock, patch
from src.llm.azure_client import AzureOpenAIClient
from src.llm.agents import AIGrowthAgent
from src.llm.chains import AnalysisChains
import pandas as pd

class TestAzureOpenAIClient:
    """测试Azure OpenAI客户端"""

    @patch('src.llm.azure_client.get_settings')
    def test_client_initialization(self, mock_settings):
        """测试客户端初始化"""
        # 模拟配置
        mock_settings.return_value.azure_openai = Mock(
            api_key='test-key',
            endpoint='https://test.openai.azure.com/',
            deployment='gpt-4',
            api_version='2024-02-15-preview'
        )

        # 创建客户端
        client = AzureOpenAIClient(temperature=0.5)

        # 验证属性
        assert client.temperature == 0.5
        assert client._llm is None  # 懒加载

    @patch('src.llm.azure_client.AzureChatOpenAI')
    @patch('src.llm.azure_client.get_settings')
    def test_generate_text(self, mock_settings, mock_llm_class):
        """测试文本生成"""
        # 设置模拟
        mock_settings.return_value.azure_openai = Mock(
            api_key='test-key',
            endpoint='https://test.openai.azure.com/',
            deployment='gpt-4',
            api_version='2024-02-15-preview'
        )

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="生成的文本")
        mock_llm_class.return_value = mock_llm

        # 测试生成
        client = AzureOpenAIClient()
        result = client.generate("测试提示词")

        # 验证调用
        assert result == "生成的文本"
        assert mock_llm.invoke.called


class TestAIGrowthAgent:
    """测试AI增长代理"""

    @patch('src.llm.agents.OrderRepository')
    @patch('src.llm.agents.CustomerRepository')
    @patch('src.llm.agents.AnalyticsRepository')
    @patch('src.llm.agents.AzureOpenAIClient')
    def test_agent_initialization(self, mock_llm, mock_analytics, mock_customer, mock_order):
        """测试代理初始化"""
        agent = AIGrowthAgent()

        # 验证工具创建
        assert len(agent.tools) > 0
        assert agent.memory is not None
        assert agent.agent is not None

    @patch('src.llm.agents.OrderRepository')
    def test_get_sales_summary_tool(self, mock_repo):
        """测试销售汇总工具"""
        # 模拟数据
        mock_repo.return_value.get_daily_sales.return_value = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=7),
            'total_revenue': [50000] * 7,
            'order_count': [800] * 7,
            'new_customer_count': [50] * 7,
            'repeat_customer_count': [750] * 7
        })

        agent = AIGrowthAgent()
        result = agent._get_sales_summary(days=7)

        # 验证结果格式
        assert "总营收" in result
        assert "总订单数" in result
        assert "销售趋势" in result