from typing import List
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .azure_client import AzureOpenAIClient
from .prompts import PromptManager


class ActionRecommendation(BaseModel):
    """行动建议模型"""
    action: str = Field(description="建议的具体行动")
    reason: str = Field(description="建议的原因")
    priority: str = Field(description="优先级：高/中/低")
    expected_impact: str = Field(description="预期效果")
    implementation_steps: List[str] = Field(description="实施步骤")


class AnalysisChains:
    """分析链集合"""

    def __init__(self):
        self.llm_client = AzureOpenAIClient()
        self.prompt_manager = PromptManager()

    def create_daily_report_chain(self) -> LLMChain:
        """创建日报生成链"""
        prompt = self.prompt_manager.get_chat_prompt("daily_summary")

        return LLMChain(
            llm=self.llm_client.llm,
            prompt=prompt,
            output_key="daily_report"
        )

    def create_problem_diagnosis_chain(self) -> SequentialChain:
        """创建问题诊断链"""
        # 第一步：分析销售下滑
        decline_analysis_prompt = self.prompt_manager.get_chat_prompt("sales_decline_analysis")
        decline_chain = LLMChain(
            llm=self.llm_client.llm,
            prompt=decline_analysis_prompt,
            output_key="decline_analysis"
        )

        # 第二步：生成行动建议
        action_prompt = ChatPromptTemplate.from_template("""
基于以下销售下滑分析：
{decline_analysis}

请生成具体的行动建议。

{format_instructions}
""")

        parser = PydanticOutputParser(pydantic_object=ActionRecommendation)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_client.llm)

        action_chain = LLMChain(
            llm=self.llm_client.llm,
            prompt=action_prompt,
            output_key="recommendations",
            output_parser=fixing_parser
        )

        # 组合成顺序链
        return SequentialChain(
            chains=[decline_chain, action_chain],
            input_variables=["period", "decline_pct", "factors", "historical_context"],
            output_variables=["decline_analysis", "recommendations"],
            verbose=True
        )

    def create_customer_insight_chain(self) -> LLMChain:
        """创建客户洞察链"""
        prompt = self.prompt_manager.get_chat_prompt("customer_insight")

        return LLMChain(
            llm=self.llm_client.llm,
            prompt=prompt,
            output_key="customer_insights"
        )