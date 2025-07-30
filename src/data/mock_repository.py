# src/data/mock_repository.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MockOrderRepository:
    """模拟订单数据仓库（用于开发和演示）"""

    def __init__(self):
        logger.info("Using mock order repository (no database connection)")
        self.db = None  # 兼容接口

    def get_daily_sales(self, start_date: datetime, end_date: datetime,
                        location_id: Optional[str] = None) -> pd.DataFrame:
        """生成模拟的每日销售数据"""
        logger.info(f"Generating mock daily sales data from {start_date} to {end_date}")

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []

        for date in dates:
            # 周末销售更高
            is_weekend = date.weekday() >= 5
            base_revenue = 50000 if is_weekend else 35000

            # 添加一些随机波动
            revenue = base_revenue + np.random.normal(0, 5000)
            orders = 800 if is_weekend else 600
            orders = int(orders + np.random.normal(0, 50))

            data.append({
                'date': date.date(),
                'location_id': location_id or 'STORE_001',
                'location_name': '示例门店',
                'order_count': max(orders, 100),
                'customer_count': int(orders * 0.8),
                'total_revenue': max(revenue, 1000),
                'avg_order_value': revenue / orders if orders > 0 else 60,
                'new_customer_count': int(orders * 0.1),
                'repeat_customer_count': int(orders * 0.7)
            })

        return pd.DataFrame(data)

    def get_item_performance(self, days: int = 30, top_n: Optional[int] = None) -> pd.DataFrame:
        """生成模拟的商品表现数据"""
        logger.info(f"Generating mock item performance data for {days} days")

        items = [
            ('珍珠奶茶', '饮品', 15, 2000),
            ('原味奶茶', '饮品', 12, 1800),
            ('烤肉饭', '主食', 35, 1500),
            ('照烧鸡饭', '主食', 32, 1400),
            ('炸鸡块', '小食', 18, 1200),
            ('薯条', '小食', 12, 1000),
            ('可乐', '饮品', 8, 900),
            ('沙拉', '轻食', 25, 600),
            ('汉堡', '主食', 28, 800),
            ('冰淇淋', '甜品', 10, 500),
            ('咖啡', '饮品', 20, 700),
            ('果汁', '饮品', 15, 650),
            ('三明治', '轻食', 22, 550),
            ('披萨', '主食', 45, 400),
            ('寿司', '主食', 38, 450)
        ]

        data = []
        for i, (item_name, category, price, base_sales) in enumerate(items):
            # 添加随机变化
            sales_variation = np.random.uniform(0.8, 1.2)
            units_sold = int(base_sales * sales_variation)

            data.append({
                'item_id': f'ITEM_{i + 1:03d}',
                'item_name': item_name,
                'category_name': category,
                'order_count': int(units_sold / 2),
                'units_sold': units_sold,
                'revenue': price * units_sold,
                'avg_price': price,
                'total_discount': price * units_sold * 0.1,
                'unique_buyers': int(units_sold / 3)
            })

        df = pd.DataFrame(data)
        if top_n:
            return df.nlargest(top_n, 'revenue')
        return df.sort_values('revenue', ascending=False)

    def get_order_items(self, start_date: datetime, end_date: datetime, location_id: Optional[str] = None) -> List[Any]:
        """生成模拟的订单项数据"""
        # 简单返回空列表，主要用于兼容接口
        return []


class MockCustomerRepository:
    """模拟客户数据仓库"""

    def __init__(self):
        logger.info("Using mock customer repository (no database connection)")
        self.db = None  # 兼容接口

    def get_customer_segments(self) -> pd.DataFrame:
        """生成模拟的客户分群数据"""
        logger.info("Generating mock customer segments data")

        segments = [
            ('高价值客户', 500, 2500000, 5000, 50),
            ('重点发展客户', 1500, 3000000, 2000, 30),
            ('普通客户', 5000, 5000000, 1000, 20),
            ('重点挽回客户', 800, 800000, 1000, 15),
            ('其他', 1200, 600000, 500, 10)
        ]

        data = []
        for segment, count, revenue, avg_value, avg_orders in segments:
            data.append({
                'segment': segment,
                'customer_count': count,
                'total_revenue': revenue,
                'avg_order_value': avg_value,
                'avg_order_count': avg_orders
            })

        return pd.DataFrame(data)

    def get_customer_profile(self, customer_id: str) -> Optional[Any]:
        """获取单个客户画像（模拟）"""
        # 返回None，主要用于兼容接口
        return None

    def get_churned_customers(self, days_threshold: int = 30) -> pd.DataFrame:
        """生成模拟的流失客户数据"""
        logger.info(f"Generating mock churned customers data (threshold: {days_threshold} days)")

        # 生成一些模拟的流失客户
        data = []
        for i in range(50):
            data.append({
                'customer_id': f'CUST_{1000 + i:04d}',
                'given_name': f'Customer{i}',
                'email_address': f'customer{i}@example.com',
                'phone_number': f'1234567{i:04d}',
                'order_final_total_cnt': np.random.randint(5, 50),
                'order_final_total_amt': np.random.randint(500, 5000),
                'order_final_avg_amt': np.random.randint(50, 150),
                'last_order_date': datetime.now() - timedelta(days=days_threshold + np.random.randint(1, 30)),
                'days_since_last_order': days_threshold + np.random.randint(1, 30),
                'high_value_customer': 1 if i < 10 else 0,
                'churned': 1
            })

        return pd.DataFrame(data)


class MockAnalyticsRepository:
    """模拟分析数据仓库"""

    def __init__(self):
        logger.info("Using mock analytics repository (no database connection)")
        self.db = None  # 兼容接口

    def get_time_series_data(self, days: int = 90) -> pd.DataFrame:
        """生成模拟的时间序列数据（用于Prophet预测）"""
        logger.info(f"Generating mock time series data for {days} days")

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # 生成带有趋势和季节性的数据
        trend = np.linspace(40000, 50000, days)  # 上升趋势
        seasonal = 5000 * np.sin(np.arange(days) * 2 * np.pi / 7)  # 周季节性
        noise = np.random.normal(0, 2000, days)  # 随机噪声

        data = pd.DataFrame({
            'ds': dates,  # Prophet需要的日期列名
            'y': trend + seasonal + noise,  # Prophet需要的目标列名
            'order_count': np.random.poisson(700, days),
            'customer_count': np.random.poisson(500, days),
            'avg_order_value': 60 + np.random.normal(0, 5, days),
            'new_customers': np.random.poisson(50, days)
        })

        # 确保销售额为正
        data['y'] = data['y'].clip(lower=10000)

        return data

    def get_promotion_effectiveness(self, days: int = 30) -> pd.DataFrame:
        """生成模拟的促销效果数据"""
        logger.info(f"Generating mock promotion effectiveness data for {days} days")

        campaigns = ['午餐优惠', '周末特惠', '会员日', '新品推广', '满减活动']

        data = []
        for campaign in campaigns:
            orders = np.random.randint(100, 500)
            revenue = orders * np.random.randint(50, 100)
            discount = revenue * np.random.uniform(0.1, 0.3)

            data.append({
                'date': datetime.now() - timedelta(days=np.random.randint(1, days)),
                'campaign': campaign,
                'order_count': orders,
                'customer_count': int(orders * 0.8),
                'total_revenue': revenue,
                'total_discount': discount,
                'avg_order_value': revenue / orders if orders > 0 else 0,
                'discount_rate': discount / revenue if revenue > 0 else 0
            })

        return pd.DataFrame(data)