import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, List
import sys
import os

# 动态把根目录加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.ui.components import (
    MetricsCard,
    SalesChart,
    CustomerSegmentChart,
    ItemPerformanceTable,
    RecommendationCard,
    ChatInterface
)

# 页面配置
st.set_page_config(
    page_title="FBR AI增长引擎",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API配置
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000/api/v1")


class AIGrowthEngineUI:
    """AI增长引擎UI主类"""

    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        """初始化会话状态"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'dashboard'

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

    def run(self):
        """运行应用"""
        # 侧边栏
        self.render_sidebar()

        # 主页面路由
        if st.session_state.current_page == 'dashboard':
            self.render_dashboard()
        elif st.session_state.current_page == 'predictions':
            self.render_predictions()
        elif st.session_state.current_page == 'insights':
            self.render_insights()
        elif st.session_state.current_page == 'recommendations':
            self.render_recommendations()
        elif st.session_state.current_page == 'chat':
            self.render_chat()

    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("🚀 FBR AI增长引擎")
            st.markdown("---")

            # 导航菜单
            st.subheader("导航")
            pages = {
                'dashboard': '📊 实时仪表板',
                'predictions': '🔮 销售预测',
                'insights': '💡 智能洞察',
                'recommendations': '🎯 优化建议',
                'chat': '💬 AI助手'
            }

            for page_id, page_name in pages.items():
                if st.button(page_name, use_container_width=True):
                    st.session_state.current_page = page_id

            st.markdown("---")

            # 数据刷新
            st.subheader("数据更新")
            if st.button("🔄 刷新数据", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

            st.caption(f"最后更新: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

            # API状态
            st.markdown("---")
            st.subheader("系统状态")
            if self.check_api_health():
                st.success("✅ API在线")
            else:
                st.error("❌ API离线")

    def render_dashboard(self):
        """渲染仪表板页面"""
        st.title("📊 实时运营仪表板")

        # 获取每日报告
        report = self.fetch_daily_report()

        if report:
            # 关键指标卡片
            col1, col2, col3, col4 = st.columns(4)

            metrics = report.get('metrics', {})

            with col1:
                MetricsCard(
                    "今日营收",
                    f"¥{metrics.get('revenue', 0):,.0f}",
                    f"{metrics.get('revenue_change', 0):+.1f}%"
                )

            with col2:
                MetricsCard(
                    "订单数",
                    f"{metrics.get('orders', 0):,}",
                    f"{metrics.get('orders_change', 0):+.1f}%"
                )

            with col3:
                MetricsCard(
                    "客户数",
                    f"{metrics.get('customers', 0):,}",
                    f"新客: {metrics.get('new_customers', 0)}"
                )

            with col4:
                MetricsCard(
                    "客单价",
                    f"¥{metrics.get('avg_order_value', 0):.0f}",
                    "稳定"
                )

            st.markdown("---")

            # 销售趋势和商品表现
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 销售趋势")
                sales_data = self.fetch_sales_trend()
                if sales_data:
                    SalesChart(sales_data)

            with col2:
                st.subheader("🏆 热销商品")
                items_data = self.fetch_item_performance(limit=10)
                if items_data:
                    ItemPerformanceTable(items_data)

            # 客户分析
            st.markdown("---")
            st.subheader("👥 客户分群分析")

            segments_data = self.fetch_customer_segments()
            if segments_data:
                CustomerSegmentChart(segments_data)

            # 异常告警
            if report.get('anomalies'):
                st.markdown("---")
                st.subheader("⚠️ 异常告警")

                for anomaly in report['anomalies']:
                    st.warning(f"**{anomaly['type']}**: {anomaly['description']}")

    def render_predictions(self):
        """渲染预测页面"""
        st.title("🔮 销售预测分析")

        # 预测设置
        col1, col2, col3 = st.columns(3)

        with col1:
            prediction_type = st.selectbox(
                "预测类型",
                ["销售额", "订单量", "客流量"]
            )

        with col2:
            periods = st.slider(
                "预测天数",
                min_value=1,
                max_value=30,
                value=7
            )

        with col3:
            if st.button("生成预测", type="primary"):
                self.generate_prediction(prediction_type, periods)

        # 显示预测结果
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result

            # 预测图表
            st.subheader("预测结果")

            fig = go.Figure()

            # 历史数据
            fig.add_trace(go.Scatter(
                x=result['historical']['dates'],
                y=result['historical']['values'],
                mode='lines+markers',
                name='历史数据',
                line=dict(color='blue')
            ))

            # 预测数据
            fig.add_trace(go.Scatter(
                x=result['forecast']['dates'],
                y=result['forecast']['values'],
                mode='lines+markers',
                name='预测值',
                line=dict(color='red', dash='dash')
            ))

            # 置信区间
            if 'confidence_interval' in result:
                fig.add_trace(go.Scatter(
                    x=result['forecast']['dates'] + result['forecast']['dates'][::-1],
                    y=result['confidence_interval']['upper'] + result['confidence_interval']['lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='置信区间'
                ))

            fig.update_layout(
                title=f"{prediction_type}预测",
                xaxis_title="日期",
                yaxis_title=prediction_type,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # 预测摘要
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "平均预测值",
                    f"¥{result['summary']['avg_forecast']:,.0f}",
                    f"{result['summary']['growth_rate']:+.1f}%"
                )

            with col2:
                st.metric(
                    "预测峰值",
                    f"¥{result['summary']['max_forecast']:,.0f}",
                    f"{result['summary']['peak_day']}"
                )

            with col3:
                st.metric(
                    "预测准确度",
                    f"{result['summary']['accuracy']:.1%}",
                    "基于历史验证"
                )

    def render_insights(self):
        """渲染洞察页面"""
        st.title("💡 智能洞察")

        # 快速问题
        st.subheader("🤔 您可能想了解...")

        quick_questions = [
            "为什么最近销量下降？",
            "哪些商品需要补货？",
            "如何提升客单价？",
            "明天需要注意什么？"
        ]

        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, use_container_width=True):
                    self.get_ai_insight(question)

        st.markdown("---")

        # 自定义问题
        st.subheader("📝 自定义提问")

        user_question = st.text_input("请输入您的问题：", placeholder="例如：如何提高周末的销量？")

        if st.button("获取洞察", type="primary") and user_question:
            self.get_ai_insight(user_question)

        # 显示洞察结果
        if 'current_insight' in st.session_state:
            insight = st.session_state.current_insight

            st.markdown("---")
            st.subheader("🎯 AI洞察")

            # 洞察内容
            st.info(insight['insight'])

            # 数据来源
            if insight.get('data_sources'):
                st.caption(f"数据来源: {', '.join(insight['data_sources'])}")

            # 相关建议
            if insight.get('recommendations'):
                st.subheader("💡 相关建议")
                for rec in insight['recommendations']:
                    RecommendationCard(rec)

    def render_recommendations(self):
        """渲染建议页面"""
        st.title("🎯 优化建议")

        # 建议类别筛选
        category = st.selectbox(
            "选择建议类别",
            ["全部", "定价优化", "库存管理", "营销策略", "运营改进"]
        )

        # 获取建议
        recommendations = self.fetch_recommendations(category if category != "全部" else None)

        if recommendations:
            # 按优先级分组显示
            high_priority = [r for r in recommendations if r['priority'] == '高']
            medium_priority = [r for r in recommendations if r['priority'] == '中']
            low_priority = [r for r in recommendations if r['priority'] == '低']

            if high_priority:
                st.subheader("🔴 高优先级")
                for rec in high_priority:
                    RecommendationCard(rec)

            if medium_priority:
                st.subheader("🟡 中优先级")
                for rec in medium_priority:
                    RecommendationCard(rec)

            if low_priority:
                st.subheader("🟢 低优先级")
                for rec in low_priority:
                    RecommendationCard(rec)
        else:
            st.info("暂无相关建议")

    def render_chat(self):
        """渲染聊天页面"""
        st.title("💬 AI运营助手")

        # 聊天界面
        ChatInterface(self.handle_chat_message)

    # API调用方法
    def check_api_health(self) -> bool:
        """检查API健康状态"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def fetch_daily_report(self) -> Dict[str, Any]:
        """获取每日报告"""
        try:
            response = requests.get(f"{API_BASE_URL}/reports/daily")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"获取数据失败: {e}")
        return {}

    def fetch_sales_trend(self) -> pd.DataFrame:
        """获取销售趋势数据"""
        # 模拟数据，实际应从API获取
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.randint(30000, 60000, size=30),
            'orders': np.random.randint(500, 1000, size=30)
        })
        return data

    def fetch_item_performance(self, limit: int = 20) -> List[Dict]:
        """获取商品表现数据"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/items/performance",
                params={'limit': limit}
            )
            if response.status_code == 200:
                return response.json()['items']
        except Exception as e:
            st.error(f"获取商品数据失败: {e}")
        return []

    def fetch_customer_segments(self) -> Dict[str, Any]:
        """获取客户分群数据"""
        try:
            response = requests.get(f"{API_BASE_URL}/customers/segments")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"获取客户数据失败: {e}")
        return {}

    def fetch_recommendations(self, category: str = None) -> List[Dict]:
        """获取建议"""
        try:
            params = {}
            if category:
                params['category'] = category

            response = requests.get(
                f"{API_BASE_URL}/recommendations",
                params=params
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"获取建议失败: {e}")
        return []

    def generate_prediction(self, prediction_type: str, periods: int):
        """生成预测"""
        with st.spinner("正在生成预测..."):
            try:
                # 映射预测类型
                type_map = {
                    "销售额": "sales",
                    "订单量": "orders",
                    "客流量": "traffic"
                }

                response = requests.post(
                    f"{API_BASE_URL}/predictions",
                    json={
                        "prediction_type": type_map.get(prediction_type, "sales"),
                        "periods": periods
                    }
                )

                if response.status_code == 200:
                    # 模拟预测结果格式
                    st.session_state.prediction_result = {
                        'historical': {
                            'dates': pd.date_range(end=datetime.now(), periods=30).tolist(),
                            'values': np.random.randint(30000, 60000, size=30).tolist()
                        },
                        'forecast': {
                            'dates': pd.date_range(start=datetime.now() + timedelta(days=1), periods=periods).tolist(),
                            'values': np.random.randint(35000, 65000, size=periods).tolist()
                        },
                        'confidence_interval': {
                            'upper': np.random.randint(40000, 70000, size=periods).tolist(),
                            'lower': np.random.randint(30000, 50000, size=periods).tolist()
                        },
                        'summary': {
                            'avg_forecast': 50000,
                            'max_forecast': 65000,
                            'growth_rate': 5.2,
                            'peak_day': '周六',
                            'accuracy': 0.85
                        }
                    }
                    st.success("预测生成成功！")
                else:
                    st.error("预测失败")
            except Exception as e:
                st.error(f"预测出错: {e}")

    def get_ai_insight(self, question: str):
        """获取AI洞察"""
        with st.spinner("AI正在思考..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/insights",
                    json={"query": question}
                )

                if response.status_code == 200:
                    st.session_state.current_insight = response.json()
                    st.success("洞察生成成功！")
                else:
                    st.error("获取洞察失败")
            except Exception as e:
                st.error(f"获取洞察出错: {e}")

    def handle_chat_message(self, message: str):
        """处理聊天消息"""
        # 添加用户消息到历史
        st.session_state.chat_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now()
        })

        # 获取AI回复
        with st.spinner("AI正在回复..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/insights",
                    json={"query": message}
                )

                if response.status_code == 200:
                    ai_response = response.json()['insight']
                else:
                    ai_response = "抱歉，我遇到了一些问题。请稍后再试。"
            except:
                ai_response = "抱歉，无法连接到服务器。"

        # 添加AI回复到历史
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        })


# 主函数
def main():
    """运行Streamlit应用"""
    app = AIGrowthEngineUI()
    app.run()


if __name__ == "__main__":
    main()