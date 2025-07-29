import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from .connectors import ClickHouseConnector
from .models import OrderItem, Customer, DailySales

logger = logging.getLogger(__name__)


class BaseRepository:
    """基础数据仓库类"""

    def __init__(self):
        self.db = ClickHouseConnector()


class OrderRepository(BaseRepository):
    """订单数据仓库"""

    def get_order_items(
            self,
            start_date: datetime,
            end_date: datetime,
            location_id: Optional[str] = None
    ) -> List[OrderItem]:
        """获取订单商品数据"""
        query = f"""
        SELECT 
            order_id,
            order_item_variation_id,
            item_variation_id,
            item_variation_name,
            item_id,
            item_name,
            category_name,
            location_id,
            location_name,
            customer_id,
            item_qty,
            item_amt,
            item_total_amt,
            item_discount,
            created_at_pt,
            campaign_names
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= '{start_date}'
            AND created_at_pt < '{end_date}'
            AND pay_status = 'COMPLETED'
        """

        params = {
            'start_date': start_date,
            'end_date': end_date
        }

        if location_id:
            query += f" AND location_id = '{location_id}'"
            params['location_id'] = location_id

        query += " ORDER BY created_at_pt DESC"

        df = self.db.execute_df(query, params)

        # 转换为模型对象
        items = []
        for _, row in df.iterrows():
            items.append(OrderItem(**row.to_dict()))

        return items

    def get_daily_sales(
            self,
            start_date: datetime,
            end_date: datetime,
            location_id: Optional[str] = None
    ) -> pd.DataFrame:
        """获取每日销售汇总"""
        query = f"""
        SELECT 
            toDate(created_at_pt) as date,
            location_id,
            COUNT(DISTINCT order_id) as order_count,
            COUNT(DISTINCT customer_id) as customer_count,
            SUM(item_total_amt) as total_revenue,
            AVG(item_total_amt) as avg_order_value,
            COUNT(DISTINCT CASE WHEN has(is_new_users, 1) THEN customer_id END) as new_customer_count,
            COUNT(DISTINCT customer_id) - COUNT(DISTINCT CASE WHEN has(is_new_users, 1) THEN customer_id END) as repeat_customer_count
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= '{start_date}'
            AND created_at_pt < '{end_date}'
            AND pay_status = 'COMPLETED'
        """

        params = {
            'start_date': start_date,
            'end_date': end_date
        }

        if location_id:
            query += f" AND location_id = '{location_id}'"
            params['location_id'] = location_id

        query += " GROUP BY date, location_id ORDER BY date DESC"

        return self.db.execute_df(query, params)

    def get_item_performance(
            self,
            days: int = 30,
            top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """获取商品销售表现"""
        query = f"""
        SELECT 
            item_id,
            item_name,
            category_name,
            COUNT(DISTINCT order_id) as order_count,
            SUM(item_qty) as units_sold,
            SUM(item_total_amt) as revenue,
            AVG(item_amt) as avg_price,
            SUM(item_discount) as total_discount,
            COUNT(DISTINCT customer_id) as unique_buyers
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= today() - {days}
            AND pay_status = 'COMPLETED'
            AND item_name IS NOT NULL
        GROUP BY item_id, item_name, category_name
        ORDER BY revenue DESC
        """

        if top_n:
            query += f" LIMIT {top_n}"

        return self.db.execute_df(query, {'days': days})


class CustomerRepository(BaseRepository):
    """客户数据仓库"""

    def get_customer_profile(self, customer_id: str) -> Optional[Customer]:
        """获取单个客户画像"""
        query = f"""
        SELECT *
        FROM ads.customer_profile
        WHERE customer_id = '{customer_id}'
        """

        df = self.db.execute_df(query, {'customer_id': customer_id})

        if len(df) > 0:
            return Customer(**df.iloc[0].to_dict())
        return None

    def get_customer_segments(self) -> pd.DataFrame:
        """获取客户分群统计"""
        query = """
        SELECT 
            CASE 
                WHEN high_value_customer = 1 THEN '高价值客户'
                WHEN key_development_customer = 1 THEN '重点发展客户'
                WHEN regular_customer = 1 THEN '普通客户'
                WHEN critical_win_back_customer = 1 THEN '重点挽回客户'
                ELSE '其他'
            END as segment,
            COUNT(*) as customer_count,
            SUM(order_final_total_amt) as total_revenue,
            AVG(order_final_avg_amt) as avg_order_value,
            AVG(order_final_total_cnt) as avg_order_count
        FROM ads.customer_profile
        WHERE order_final_total_cnt > 0
        GROUP BY segment
        ORDER BY total_revenue DESC
        """

        return self.db.execute_df(query)

    def get_churned_customers(self, days_threshold: int = 30) -> pd.DataFrame:
        """获取流失客户名单"""
        query = f"""
        WITH customer_last_order AS (
            SELECT 
                customer_id,
                MAX(toDate(created_at_pt)) as last_order_date
            FROM dw.fact_order_item_variations
            WHERE pay_status = 'COMPLETED'
            GROUP BY customer_id
        )
        SELECT 
            cp.*,
            clo.last_order_date,
            dateDiff('day', clo.last_order_date, today()) as days_since_last_order
        FROM ads.customer_profile cp
        JOIN customer_last_order clo ON cp.customer_id = clo.customer_id
        WHERE dateDiff('day', clo.last_order_date, today()) > {days_threshold}
            AND cp.order_final_total_cnt > 3  -- 曾经的活跃客户
        ORDER BY cp.order_final_total_amt DESC
        """

        return self.db.execute_df(query, {'days_threshold': days_threshold})


class AnalyticsRepository(BaseRepository):
    """分析数据仓库"""

    def get_time_series_data(self, days: int = 90) -> pd.DataFrame:
        """获取时间序列数据用于预测"""
        query = f"""
        SELECT 
            toDate(created_at_pt) as ds,  -- Prophet需要的列名
            SUM(item_total_amt) as y,      -- Prophet需要的列名
            COUNT(DISTINCT order_id) as order_count,
            COUNT(DISTINCT customer_id) as customer_count,
            AVG(item_total_amt) as avg_order_value,
            SUM(CASE WHEN has(is_new_users, 1) THEN 1 ELSE 0 END) as new_customers
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= today() - {days}
            AND pay_status = 'COMPLETED'
        GROUP BY ds
        ORDER BY ds
        """

        return self.db.execute_df(query, {'days': days})

    def get_promotion_effectiveness(self, days: int = 30) -> pd.DataFrame:
        """获取促销效果数据"""
        query = f"""
        WITH promotion_orders AS (
            SELECT 
                toDate(created_at_pt) as date,
                arrayJoin(campaign_names) as campaign,
                order_id,
                item_total_amt,
                item_discount,
                customer_id
            FROM dw.fact_order_item_variations
            WHERE created_at_pt >= today() - {days}
                AND pay_status = 'COMPLETED'
                AND length(campaign_names) > 0
        )
        SELECT 
            campaign,
            COUNT(DISTINCT order_id) as order_count,
            COUNT(DISTINCT customer_id) as customer_count,
            SUM(item_total_amt) as total_revenue,
            SUM(item_discount) as total_discount,
            AVG(item_total_amt) as avg_order_value,
            SUM(item_discount) / SUM(item_total_amt + item_discount) as discount_rate
        FROM promotion_orders
        GROUP BY campaign
        ORDER BY total_revenue DESC
        """

        return self.db.execute_df(query, {'days': days})