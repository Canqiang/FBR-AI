import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from .azure_client import AzureOpenAIClient
from ..data.repositories import OrderRepository, CustomerRepository, AnalyticsRepository

logger = logging.getLogger(__name__)


# 工具输入模型
class DateRangeInput(BaseModel):
    """日期范围输入"""
    days: int = Field(description="查询最近多少天的数据", default=7)


class CustomerAnalysisInput(BaseModel):
    """客户分析输入"""
    segment: Optional[str] = Field(description="客户分群类型", default=None)


class ItemAnalysisInput(BaseModel):
    """商品分析输入"""
    category: Optional[str] = Field(description="商品类别", default=None)
    top_n: int = Field(description="返回前N个商品", default=10)


class AIGrowthAgent:
    """AI增长引擎智能代理"""

    def __init__(self):
        self.llm_client = AzureOpenAIClient(temperature=0.7)
        self.order_repo = OrderRepository()
        self.customer_repo = CustomerRepository()
        self.analytics_repo = AnalyticsRepository()

        # 初始化工具
        self.tools = self._create_tools()

        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 创建代理
        self.agent = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """创建代理工具"""
        tools = [
            StructuredTool(
                name="get_sales_summary",
                description="获取销售汇总数据，包括订单数、营收、客单价等",
                func=self._get_sales_summary,
                args_schema=DateRangeInput
            ),
            StructuredTool(
                name="analyze_customer_segments",
                description="分析客户分群，了解不同类型客户的价值和行为",
                func=self._analyze_customer_segments,
                args_schema=CustomerAnalysisInput
            ),
            StructuredTool(
                name="get_item_performance",
                description="获取商品销售表现，找出热销和滞销商品",
                func=self._get_item_performance,
                args_schema=ItemAnalysisInput
            ),
            Tool(
                name="get_churned_customers",
                description="获取流失客户名单和特征",
                func=self._get_churned_customers
            ),
            Tool(
                name="get_promotion_effectiveness",
                description="分析促销活动效果",
                func=self._get_promotion_effectiveness
            )
        ]

        return tools

    def _create_agent(self) -> AgentExecutor:
        """创建智能代理"""
        # 代理提示词 - 修复：添加 {tools} 占位符
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是FBR餐饮AI增长引擎的智能助手。你可以：
    1. 分析销售数据，发现问题和机会
    2. 识别客户行为模式，提供营销建议
    3. 优化商品组合和定价策略
    4. 预测未来趋势，制定增长策略

    始终基于数据说话，提供具体、可执行的建议。

    你可以使用以下工具：
    {{tools}}

    当前对话历史：
    {chat_history}

    用户问题：{input}

    请思考后回答，如果需要，使用合适的工具获取数据。
    {agent_scratchpad}
    """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 创建 OpenAI Tools Agent
        try:
            agent = create_openai_tools_agent(
                llm=self.llm_client.llm,
                tools=self.tools,
                prompt=prompt
            )

            # 创建代理执行器
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )

            return agent_executor
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

    def chat(self, query: str) -> Dict[str, Any]:
        """与代理对话"""
        try:
            logger.info(f"Agent chat query: {query}")

            # 调用代理
            result = self.agent.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages
            })

            return {
                'answer': result.get('output', '抱歉，我无法理解您的问题'),
                'intermediate_steps': result.get('intermediate_steps', [])
            }
        except Exception as e:
            logger.error(f"Agent chat failed: {str(e)}")
            return {
                'answer': f'抱歉，处理您的请求时出现错误。错误信息：{str(e)}',
                'intermediate_steps': []
            }

    # 工具实现
    # 工具实现方法
    def _get_sales_summary(self, days: int = 7) -> str:
        """获取销售汇总数据"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            df = self.order_repo.get_daily_sales(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if df.empty:
                return "没有找到销售数据"

            # 计算汇总指标
            total_revenue = df['total_revenue'].sum()
            total_orders = df['order_count'].sum()
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            total_customers = df['customer_count'].sum()

            result = f"""
    最近{days}天销售汇总：
    - 总营收：¥{total_revenue:,.0f}
    - 总订单数：{total_orders:,}
    - 平均客单价：¥{avg_order_value:.0f}
    - 总客户数：{total_customers:,}
    - 新客户数：{df['new_customer_count'].sum():,}
    - 复购客户数：{df['repeat_customer_count'].sum():,}
    """
            return result
        except Exception as e:
            logger.error(f"Error getting sales summary: {e}")
            return f"获取销售数据时出错：{str(e)}"

    def _analyze_customer_segments(self, segment: Optional[str] = None) -> str:
        """分析客户分群"""
        try:
            df = self.customer_repo.get_customer_segments()

            if df.empty:
                return "没有客户分群数据"

            result = "客户分群分析：\n"

            for _, row in df.iterrows():
                if segment and row.get('segment', '') != segment:
                    continue

                result += f"\n{row.get('segment', 'Unknown')}：\n"
                result += f"- 客户数量：{row.get('customer_count', 0):,}\n"
                result += f"- 总贡献营收：¥{row.get('total_revenue', 0):,.0f}\n"
                result += f"- 平均客单价：¥{row.get('avg_order_value', 0):.0f}\n"
                result += f"- 平均订单数：{row.get('avg_order_count', 0):.0f}\n"

            return result
        except Exception as e:
            logger.error(f"Error analyzing customer segments: {e}")
            return f"分析客户分群时出错：{str(e)}"

    def _get_item_performance(self, category: Optional[str] = None, top_n: int = 10) -> str:
        """获取商品表现"""
        try:
            df = self.order_repo.get_item_performance(days=30, top_n=top_n)

            if df.empty:
                return "没有商品销售数据"

            if category:
                df = df[df.get('category_name', '') == category]

            result = f"商品销售表现TOP{min(len(df), top_n)}：\n"

            for i, (_, row) in enumerate(df.head(top_n).iterrows()):
                result += f"\n{i + 1}. {row.get('item_name', 'Unknown')}\n"
                result += f"   - 类别：{row.get('category_name', 'Unknown')}\n"
                result += f"   - 销售额：¥{row.get('revenue', 0):,.0f}\n"
                result += f"   - 销售量：{row.get('units_sold', 0):,}件\n"
                result += f"   - 平均价格：¥{row.get('avg_price', 0):.0f}\n"

            return result
        except Exception as e:
            logger.error(f"Error getting item performance: {e}")
            return f"获取商品表现时出错：{str(e)}"

    def _get_churned_customers(self) -> str:
        """获取流失客户"""
        try:
            # 这里可以调用实际的流失客户分析方法
            return "流失客户分析功能正在开发中..."
        except Exception as e:
            return f"分析流失客户时出错：{str(e)}"

    def _get_promotion_effectiveness(self) -> str:
        """分析促销效果"""
        try:
            # 这里可以调用实际的促销效果分析方法
            return "促销效果分析功能正在开发中..."
        except Exception as e:
            return f"分析促销效果时出错：{str(e)}"