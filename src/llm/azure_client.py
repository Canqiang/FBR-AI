import logging
from typing import Optional, List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LoggingCallbackHandler(BaseCallbackHandler):
    """LLM调用日志回调处理器"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        logger.debug(f"LLM Start - Prompts: {prompts[:100]}...")

    def on_llm_end(self, response, **kwargs):
        logger.debug(f"LLM End - Response: {str(response)[:100]}...")

    def on_llm_error(self, error: Exception, **kwargs):
        logger.error(f"LLM Error: {error}")


class AzureOpenAIClient:
    """Azure OpenAI客户端封装"""

    def __init__(self, temperature: float = 0.0):
        self.settings = get_settings().azure_openai
        self.temperature = temperature
        self._llm = None

    @property
    def llm(self) -> AzureChatOpenAI:
        """获取LLM实例（懒加载）"""
        if self._llm is None:
            self._llm = AzureChatOpenAI(
                deployment_name=self.settings.deployment,
                openai_api_key=self.settings.api_key,
                openai_api_version=self.settings.api_version,
                openai_api_base=self.settings.endpoint,
                temperature=self.temperature,
                callbacks=[LoggingCallbackHandler()]
            )
            logger.info(f"Azure OpenAI client initialized with deployment: {self.settings.deployment}")
        return self._llm

    def chat(self, messages: List[BaseMessage]) -> str:
        """发送聊天消息"""
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        return self.chat(messages)