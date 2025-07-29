import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from clickhouse_driver import Client
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """ClickHouse数据库连接器"""

    def __init__(self):
        self.settings = get_settings().clickhouse
        self._client = None

    @property
    def client(self) -> Client:
        """获取客户端实例（懒加载）"""
        if self._client is None:
            self._client = Client(
                host=self.settings.host,
                port=self.settings.port,
                database=self.settings.database,
                user=self.settings.user,
                password=self.settings.password
            )
            logger.info(f"Connected to ClickHouse: {self.settings.host}:{self.settings.port}")
        return self._client

    def execute(self, query: str, params: Dict[str, Any] = None) -> List[tuple]:
        """执行查询"""
        try:
            logger.debug(f"Executing query: {query[:100]}...")
            return self.client.execute(query, params or {})
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_df(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        try:
            result = self.execute(query, params)
            if result and len(result) > 0:
                # 获取列名
                columns = self.client.execute(f"DESCRIBE ({query})")
                column_names = [col[0] for col in columns]
                return pd.DataFrame(result, columns=column_names[:len(result[0])])
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to convert to DataFrame: {e}")
            raise

    def insert_df(self, table: str, df: pd.DataFrame, batch_size: int = 10000):
        """批量插入DataFrame数据"""
        try:
            total_rows = len(df)
            logger.info(f"Inserting {total_rows} rows into {table}")

            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                self.client.execute(
                    f"INSERT INTO {table} VALUES",
                    batch.to_dict('records')
                )
                logger.debug(f"Inserted batch {i // batch_size + 1}")

            logger.info(f"Successfully inserted {total_rows} rows")
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.disconnect()
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