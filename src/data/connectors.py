import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import clickhouse_connect
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """ClickHouse数据库连接器（基于clickhouse_connect）"""

    def __init__(self):
        self.settings = get_settings().clickhouse
        self._client = None

    @property
    def client(self):
        """获取客户端实例（懒加载）"""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.settings.host,
                port=self.settings.port,
                database=self.settings.database,
                username=self.settings.user,
                password=self.settings.password,
                secure=(self.settings.port == 443 or self.settings.secure),  # 443端口自动secure
                verify=False  # 需要则可调整
            )
            logger.info(f"Connected to ClickHouse: {self.settings.host}:{self.settings.port}")
        return self._client

    def execute(self, query: str, params: Dict[str, Any] = None) -> List[tuple]:
        """执行查询并返回原始结果"""
        try:
            logger.debug(f"Executing query: {query[:100]}...")
            # clickhouse_connect 没有 params 语法，需手工格式化
            if params:
                query = query.format(**params)
            result = self.client.query(query)
            return result.result_rows
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_df(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        try:
            # clickhouse_connect 推荐直接用 query_df
            if params:
                query = query.format(**params)
            return self.client.query_df(query)
        except Exception as e:
            logger.error(f"Failed to convert to DataFrame: {e}")
            raise

    def insert_df(self, table: str, df: pd.DataFrame, batch_size: int = 10000):
        """批量插入DataFrame数据（高效版）"""
        try:
            total_rows = len(df)
            logger.info(f"Inserting {total_rows} rows into {table}")

            # clickhouse_connect 支持DataFrame直接插入
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                self.client.insert_df(table, batch)
                logger.debug(f"Inserted batch {i // batch_size + 1}")

            logger.info(f"Successfully inserted {total_rows} rows")
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise

    def close(self):
        """关闭连接（clickhouse_connect自动管理，通常无需手动关闭）"""
        # 可加上 self._client = None 防止重复释放
        self._client = None
        logger.info("ClickHouse connection closed")

    @contextmanager
    def transaction(self):
        """事务上下文管理器（ClickHouse不支持事务，这里仅作为接口）"""
        try:
            yield self
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            pass
