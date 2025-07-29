from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from decimal import Decimal

@dataclass
class OrderItem:
    """订单商品模型"""
    order_id: str
    order_item_variation_id: str
    item_variation_id: Optional[str]
    item_variation_name: Optional[str]
    item_id: Optional[str]
    item_name: Optional[str]
    category_name: str
    location_id: Optional[str]
    location_name: Optional[str]
    customer_id: Optional[str]
    item_qty: int
    item_amt: Decimal
    item_total_amt: Decimal
    item_discount: Decimal
    created_at_pt: datetime
    campaign_names: List[str] = field(default_factory=list)

@dataclass
class Customer:
    """客户模型"""
    customer_id: str
    given_name: Optional[str]
    family_name: Optional[str]
    email_address: Optional[str]
    phone_number: Optional[str]
    customer_created_date: Optional[datetime]
    order_final_total_cnt: int
    order_final_total_amt: Decimal
    order_final_avg_amt: float
    high_value_customer: bool
    churned: bool
    rfm_labels: List[str] = field(default_factory=list)

@dataclass
class DailySales:
    """每日销售汇总"""
    date: datetime
    location_id: Optional[str]
    order_count: int
    customer_count: int
    total_revenue: Decimal
    avg_order_value: Decimal
    new_customer_count: int
    repeat_customer_count: int