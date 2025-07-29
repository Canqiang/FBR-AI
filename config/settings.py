import os
import json
import logging
from typing import Dict, Any
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI配置"""
    api_key: str
    endpoint: str
    deployment: str
    api_version: str


@dataclass
class ClickHouseConfig:
    """ClickHouse数据库配置"""
    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass
class AppConfig:
    """应用程序配置"""
    log_level: str
    log_file: str
    weather_api_key: str


class Settings:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        self._config_path = config_path or self._get_config_path()
        self._config = self._load_config()

        # 初始化各配置对象
        self.azure_openai = self._get_azure_openai_config()
        self.clickhouse = self._get_clickhouse_config()
        self.app = self._get_app_config()

        # 设置日志
        self._setup_logging()

    def _get_config_path(self) -> str:
        """获取配置文件路径"""
        # 优先使用环境变量
        env_path = os.getenv("FBR_CONFIG_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 默认路径
        default_paths = [
            "config/config.json",
            "../config/config.json",
            "../../config/config.json"
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Config file not found. Please create config/config.json from template."
        )

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Config loaded from {self._config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _get_azure_openai_config(self) -> AzureOpenAIConfig:
        """获取Azure OpenAI配置"""
        return AzureOpenAIConfig(
            api_key=self._get_config_value("AZURE_OPENAI_API_KEY"),
            endpoint=self._get_config_value("AZURE_OPENAI_ENDPOINT"),
            deployment=self._get_config_value("AZURE_OPENAI_DEPLOYMENT"),
            api_version=self._get_config_value("AZURE_OPENAI_API_VERSION")
        )

    def _get_clickhouse_config(self) -> ClickHouseConfig:
        """获取ClickHouse配置"""
        return ClickHouseConfig(
            host=self._get_config_value("CLICKHOUSE_HOST", "localhost"),
            port=int(self._get_config_value("CLICKHOUSE_PORT", 9000)),
            database=self._get_config_value("CLICKHOUSE_DATABASE", "dw"),
            user=self._get_config_value("CLICKHOUSE_USER", "default"),
            password=self._get_config_value("CLICKHOUSE_PASSWORD", "")
        )

    def _get_app_config(self) -> AppConfig:
        """获取应用配置"""
        return AppConfig(
            log_level=self._get_config_value("LOG_LEVEL", "INFO"),
            log_file=self._get_config_value("LOG_FILE", "logs/ai_growth_engine.log"),
            weather_api_key=self._get_config_value("WEATHER_API_KEY", "")
        )

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持环境变量覆盖"""
        # 环境变量优先
        env_value = os.getenv(key)
        if env_value:
            return env_value

        # 配置文件
        if key in self._config:
            return self._config[key]

        # 默认值
        if default is not None:
            return default

        raise ValueError(f"Configuration key '{key}' not found")

    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(self.app.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.app.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.app.log_file),
                logging.StreamHandler()
            ]
        )

# 单例模式
_settings = None

def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings