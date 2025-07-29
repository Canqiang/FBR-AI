from typing import Dict, Any
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

class PromptManager:
    """Prompt模板管理器"""
    # 系统提示词
    SYSTEM_PROMPTS = {
        "analyst": """你是一位专业的餐饮业务数据分析师。你的任务是：
1. 分析餐饮业务数据，发现问题和机会
2. 提供清晰、可执行的建议
3. 用简单易懂的语言解释复杂的数据洞察
4. 始终以提升营收和客户满意度为目标

回答时请：
- 使用具体的数字和百分比
- 提供明确的行动建议
- 考虑餐饮行业的特点（如用餐高峰、季节性等）""",

        "advisor": """你是一位资深的餐饮运营顾问。基于数据分析结果，你需要：
1. 解释数据背后的业务含义
2. 提供实用的运营建议
3. 预测建议的效果
4. 考虑实施的可行性

请用餐厅老板能理解的方式沟通。"""
    }

    # 分析提示词模板
    ANALYSIS_PROMPTS = {
        "daily_summary": PromptTemplate(
            input_variables=["date", "metrics", "comparison"],
            template="""
请为{date}生成餐厅运营日报。

今日关键指标：
{metrics}

与昨日对比：
{comparison}

请生成一份简洁的日报，包括：
1. 一句话总结今日表现
2. 3个关键发现（用数据支撑）
3. 明天需要关注的重点
4. 1-2个具体的行动建议
"""
        ),

        "sales_decline_analysis": PromptTemplate(
            input_variables=["period", "decline_pct", "factors", "historical_context"],
            template="""
分析{period}的销售下滑情况。

下滑幅度：{decline_pct}%

可能的影响因素：
{factors}

历史数据参考：
{historical_context}

请分析：
1. 最可能的主要原因（给出置信度）
2. 次要影响因素
3. 是否为正常波动
4. 具体的应对措施（按优先级排序）
"""
        ),

        "customer_insight": PromptTemplate(
            input_variables=["segment_data", "behavior_patterns", "trends"],
            template="""
基于客户数据生成洞察。

客户分群数据：
{segment_data}

行为模式：
{behavior_patterns}

近期趋势：
{trends}

请提供：
1. 各客户群体的特征总结
2. 需要重点关注的客户群体及原因
3. 针对不同群体的营销建议
4. 预期效果评估
"""
        ),

        "item_recommendation": PromptTemplate(
            input_variables=["sales_data", "inventory_status", "season_factors"],
            template="""
基于商品销售数据提供建议。

销售数据：
{sales_data}

库存状况：
{inventory_status}

季节因素：
{season_factors}

请建议：
1. 需要增加备货的商品及数量
2. 建议促销的滞销商品
3. 新品开发方向
4. 定价优化建议
"""
        )
    }

    @classmethod
    def get_chat_prompt(cls, prompt_type: str, **kwargs) -> ChatPromptTemplate:
        """获取聊天提示词模板"""
        system_prompt = cls.SYSTEM_PROMPTS.get("analyst", "")
        human_prompt = cls.ANALYSIS_PROMPTS.get(prompt_type)

        if not human_prompt:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate(prompt=human_prompt)
        ])
