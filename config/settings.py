# config/settings.py
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

        # 查找配置文件
        current_dir = Path(__file__).parent
        project_root = current_dir.parent

        possible_paths = [
            current_dir / "config.json",
            project_root / "config" / "config.json",
            Path.cwd() / "config" / "config.json",
            Path.home() / ".fbr" / "config.json"
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # 如果没找到，返回默认路径
        default_path = project_root / "config" / "config.json"
        logger.warning(f"Config file not found, will use defaults. Expected at: {default_path}")
        return str(default_path)

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 默认配置
        default_config = {
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_DEPLOYMENT": "",
            "AZURE_OPENAI_API_VERSION": "",
            "CLICKHOUSE_HOST": "localhost",
            "CLICKHOUSE_PORT": 443,
            "CLICKHOUSE_DATABASE": "dw",
            "CLICKHOUSE_USER": "",
            "CLICKHOUSE_PASSWORD": "",
            "WEATHER_API_KEY": "",
            "LOG_LEVEL": "INFO",
            "LOG_FILE": "logs/ai_growth_engine.log"
        }

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                # 合并配置
                config = {**default_config, **file_config}
                logger.info(f"Config loaded from {self._config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self._config_path}, using defaults")
            return default_config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            logger.warning("Using default configuration")
            return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.warning("Using default configuration")
            return default_config

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持环境变量覆盖"""
        # 环境变量优先
        env_value = os.getenv(key)
        if env_value:
            # 对于端口号等数字类型，进行转换
            if key.endswith('_PORT') and env_value.isdigit():
                return int(env_value)
            return env_value

        # 配置文件
        if key in self._config:
            return self._config[key]

        # 默认值
        if default is not None:
            return default

        # 对于某些关键配置，返回空字符串而不是报错
        if key.startswith('AZURE_'):
            return ""

        raise ValueError(f"Configuration key '{key}' not found")

    def _get_azure_openai_config(self) -> AzureOpenAIConfig:
        """获取Azure OpenAI配置"""
        return AzureOpenAIConfig(
            api_key=self._get_config_value("AZURE_OPENAI_API_KEY", ""),
            endpoint=self._get_config_value("AZURE_OPENAI_ENDPOINT", ""),
            deployment=self._get_config_value("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            api_version=self._get_config_value("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
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

    def _setup_logging(self):
        """设置日志"""
        log_file = Path(self.app.log_file)
        log_dir = log_file.parent

        # 创建日志目录
        log_dir.mkdir(parents=True, exist_ok=True)

        # 配置日志
        logging.basicConfig(
            level=getattr(logging, self.app.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def has_azure_openai(self) -> bool:
        """检查是否配置了Azure OpenAI"""
        return bool(self.azure_openai.api_key and self.azure_openai.endpoint)

    def has_clickhouse(self) -> bool:
        """检查是否配置了ClickHouse"""
        return bool(self.clickhouse.host)


# 单例模式
_settings = None


def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings