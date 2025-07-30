# src/data/connectors.py
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import pandas as pd

logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """ClickHouse数据库连接器"""

    def __init__(self):
        from config.settings import get_settings
        self.settings = get_settings().clickhouse
        self._client = None
        self._connection_failed = False

    @property
    def client(self):
        """获取客户端实例（懒加载）"""
        if self._client is None and not self._connection_failed:
            try:
                # 尝试使用clickhouse_connect
                try:
                    import clickhouse_connect
                    self._client = clickhouse_connect.get_client(
                        host=self.settings.host,
                        port=self.settings.port,
                        username=self.settings.user,
                        password=self.settings.password,
                        database=self.settings.database,
                        secure=False  # 本地开发使用非加密连接
                    )
                    logger.info(
                        f"Connected to ClickHouse using clickhouse_connect: {self.settings.host}:{self.settings.port}")
                except ImportError:
                    # 如果没有clickhouse_connect，尝试使用clickhouse_driver
                    from clickhouse_driver import Client
                    self._client = Client(
                        host=self.settings.host,
                        port=self.settings.port,
                        database=self.settings.database,
                        user=self.settings.user,
                        password=self.settings.password
                    )
                    logger.info(
                        f"Connected to ClickHouse using clickhouse_driver: {self.settings.host}:{self.settings.port}")
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse: {e}")
                self._connection_failed = True
                raise ConnectionError(f"Cannot connect to ClickHouse: {e}")

        if self._connection_failed:
            raise ConnectionError("ClickHouse connection has failed previously")

        return self._client

    def execute(self, query: str, params: Dict[str, Any] = None) -> List[tuple]:
        """执行查询"""
        try:
            logger.debug(f"Executing query: {query[:100]}...")

            # 根据客户端类型调用不同的方法
            if hasattr(self.client, 'query'):  # clickhouse_connect
                result = self.client.query(query, parameters=params or {})
                return result.result_rows if hasattr(result, 'result_rows') else result
            else:  # clickhouse_driver
                return self.client.execute(query, params or {})
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_df(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        try:
            # 使用query_df方法（如果可用）
            if hasattr(self.client, 'query_df'):
                return self.client.query_df(query)
            else:
                # 否则手动转换
                result = self.execute(query, params)
                if result and len(result) > 0:
                    # 获取列名
                    columns = self.client.execute(f"DESCRIBE ({query})") if hasattr(self.client, 'execute') else []
                    if columns:
                        column_names = [col[0] for col in columns]
                        return pd.DataFrame(result, columns=column_names[:len(result[0])])
                    else:
                        # 无法获取列名，使用默认列名
                        return pd.DataFrame(result)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to convert to DataFrame: {e}")
            # 返回空DataFrame而不是抛出异常
            return pd.DataFrame()

    def insert_df(self, table: str, df: pd.DataFrame, batch_size: int = 10000):
        """批量插入DataFrame数据"""
        try:
            total_rows = len(df)
            logger.info(f"Inserting {total_rows} rows into {table}")

            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]

                # 根据客户端类型调用不同的方法
                if hasattr(self.client, 'insert_df'):  # clickhouse_connect
                    self.client.insert_df(table, batch)
                else:  # clickhouse_driver
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
            try:
                if hasattr(self._client, 'close'):
                    self._client.close()
                elif hasattr(self._client, 'disconnect'):
                    self._client.disconnect()
            except:
                pass
            self._client = None
            self._connection_failed = False
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