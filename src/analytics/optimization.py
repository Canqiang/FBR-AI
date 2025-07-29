import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize, linprog
from sklearn.linear_model import LinearRegression
import cvxpy as cp

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """优化引擎"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def optimize_pricing(
            self,
            data: pd.DataFrame,
            item_costs: Dict[str, float],
            constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """优化定价策略"""
        logger.info("Optimizing pricing strategy")

        # 计算价格弹性
        elasticities = self._calculate_price_elasticity(data)

        # 优化每个商品的价格
        optimal_prices = {}
        expected_revenue = {}

        for item, elasticity in elasticities.items():
            if item not in item_costs:
                continue

            cost = item_costs[item]
            current_price = data[data['item_name'] == item]['item_amt'].mean()

            # 定义收入函数：R = P * Q, Q = Q0 * (P/P0)^elasticity
            def revenue_function(price):
                quantity = 100 * (price / current_price) ** elasticity
                return -price * quantity  # 负号因为我们要最大化

            # 约束条件
            bounds = [(cost * 1.2, current_price * 2)]  # 价格范围

            if constraints and 'max_price_increase' in constraints:
                max_increase = constraints['max_price_increase']
                bounds = [(cost * 1.2, current_price * (1 + max_increase))]

            # 优化
            result = minimize(revenue_function, x0=current_price, bounds=bounds)

            optimal_price = result.x[0]
            optimal_revenue = -result.fun

            optimal_prices[item] = {
                'current_price': current_price,
                'optimal_price': optimal_price,
                'price_change_pct': (optimal_price - current_price) / current_price * 100,
                'expected_revenue': optimal_revenue,
                'elasticity': elasticity
            }

        return {
            'optimal_prices': optimal_prices,
            'total_revenue_increase': sum(p['expected_revenue'] for p in optimal_prices.values()) - data[
                'item_total_amt'].sum(),
            'recommendations': self._generate_pricing_recommendations(optimal_prices)
        }

    def _calculate_price_elasticity(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算价格弹性"""
        elasticities = {}

        for item in data['item_name'].unique():
            item_data = data[data['item_name'] == item]

            if len(item_data) < 10:
                continue

            # 按价格分组计算需求量
            price_demand = item_data.groupby('item_amt')['item_qty'].sum().reset_index()

            if len(price_demand) > 2:
                # 对数回归估计弹性
                X = np.log(price_demand['item_amt'].values.reshape(-1, 1))
                y = np.log(price_demand['item_qty'].values)

                model = LinearRegression()
                model.fit(X, y)

                elasticity = model.coef_[0]
                elasticities[item] = elasticity

        return elasticities

    def optimize_inventory(
            self,
            demand_forecast: pd.DataFrame,
            storage_capacity: int,
            item_constraints: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """优化库存分配"""
        logger.info("Optimizing inventory allocation")

        items = demand_forecast['item_name'].unique()
        n_items = len(items)

        # 预测需求
        demands = demand_forecast.groupby('item_name')['predicted_demand'].sum().values

        # 单位存储成本（假设）
        storage_costs = np.ones(n_items) * 0.1

        # 缺货成本（假设）
        shortage_costs = demands * 0.5

        # 决策变量：库存量
        inventory = cp.Variable(n_items, nonneg=True)

        # 目标：最小化总成本
        storage_cost = cp.sum(storage_costs @ inventory)
        shortage_cost = cp.sum(shortage_costs @ cp.maximum(demands - inventory, 0))
        objective = cp.Minimize(storage_cost + shortage_cost)

        # 约束条件
        constraints = [
            cp.sum(inventory) <= storage_capacity,  # 总库存约束
            inventory >= demands * 0.8,  # 最小库存约束（满足80%需求）
        ]

        # 添加商品特定约束
        if item_constraints:
            for i, item in enumerate(items):
                if item in item_constraints:
                    if 'min_stock' in item_constraints[item]:
                        constraints.append(inventory[i] >= item_constraints[item]['min_stock'])
                    if 'max_stock' in item_constraints[item]:
                        constraints.append(inventory[i] <= item_constraints[item]['max_stock'])

        # 求解
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # 整理结果
        optimal_inventory = {}
        for i, item in enumerate(items):
            optimal_inventory[item] = {
                'predicted_demand': demands[i],
                'optimal_stock': inventory.value[i],
                'stock_coverage': inventory.value[i] / demands[i] if demands[i] > 0 else 0,
                'storage_cost': storage_costs[i] * inventory.value[i],
                'expected_shortage': max(demands[i] - inventory.value[i], 0)
            }

        return {
            'optimal_inventory': optimal_inventory,
            'total_storage_cost': storage_cost.value,
            'total_shortage_cost': shortage_cost.value,
            'total_cost': objective.value,
            'capacity_utilization': sum(inventory.value) / storage_capacity
        }

    def optimize_staff_scheduling(
            self,
            demand_forecast: pd.DataFrame,
            staff_costs: Dict[str, float],
            constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """优化员工排班"""
        logger.info("Optimizing staff scheduling")

        # 时段划分（假设按小时）
        hours = range(24)
        staff_types = list(staff_costs.keys())

        # 预测每小时所需员工数
        hourly_demand = demand_forecast.groupby('hour')['staff_required'].mean().values

        # 决策变量：每种类型员工在每个时段的数量
        n_hours = len(hours)
        n_types = len(staff_types)

        # 使用线性规划
        c = []  # 成本系数
        A_ub = []  # 不等式约束矩阵
        b_ub = []  # 不等式约束向量

        # 构建成本向量
        for staff_type in staff_types:
            c.extend([staff_costs[staff_type]] * n_hours)

        # 需求约束：每小时的员工总数要满足需求
        for h in range(n_hours):
            constraint = [0] * (n_types * n_hours)
            for t in range(n_types):
                constraint[t * n_hours + h] = 1
            A_ub.append([-x for x in constraint])  # 转为 <= 形式
            b_ub.append(-hourly_demand[h])

        # 求解
        bounds = [(0, constraints.get('max_staff_per_hour', 10))] * (n_types * n_hours)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success:
            # 整理结果
            schedule = {}
            for t, staff_type in enumerate(staff_types):
                schedule[staff_type] = result.x[t * n_hours:(t + 1) * n_hours]

            return {
                'optimal_schedule': schedule,
                'total_cost': result.fun,
                'hourly_coverage': self._calculate_coverage(schedule, hourly_demand),
                'recommendations': self._generate_scheduling_recommendations(schedule, hourly_demand)
            }
        else:
            logger.error("Optimization failed")
            return {'error': 'Optimization failed', 'message': result.message}

    def _generate_pricing_recommendations(self, optimal_prices: Dict) -> List[Dict]:
        """生成定价建议"""
        recommendations = []

        for item, pricing in optimal_prices.items():
            if pricing['price_change_pct'] > 5:
                recommendations.append({
                    'item': item,
                    'action': '建议涨价',
                    'current_price': pricing['current_price'],
                    'suggested_price': pricing['optimal_price'],
                    'reason': f'需求价格弹性较低（{pricing["elasticity"]:.2f}），涨价可增加收入',
                    'expected_impact': f'预计增加收入{pricing["expected_revenue"] - pricing["current_price"] * 100:.0f}元'
                })
            elif pricing['price_change_pct'] < -5:
                recommendations.append({
                    'item': item,
                    'action': '建议降价',
                    'current_price': pricing['current_price'],
                    'suggested_price': pricing['optimal_price'],
                    'reason': f'需求价格弹性较高（{pricing["elasticity"]:.2f}），降价可增加销量',
                    'expected_impact': f'预计增加收入{pricing["expected_revenue"] - pricing["current_price"] * 100:.0f}元'
                })

        return recommendations

    def _calculate_coverage(self, schedule: Dict, demand: np.ndarray) -> List[float]:
        """计算人员覆盖率"""
        coverage = []
        for h in range(len(demand)):
            total_staff = sum(schedule[staff_type][h] for staff_type in schedule)
            coverage.append(total_staff / demand[h] if demand[h] > 0 else 1.0)
        return coverage

    def _generate_scheduling_recommendations(self, schedule: Dict, demand: np.ndarray) -> List[str]:
        """生成排班建议"""
        recommendations = []

        # 找出高峰时段
        peak_hours = np.where(demand > np.percentile(demand, 75))[0]
        recommendations.append(f"高峰时段（{peak_hours.tolist()}）需要加强人员配备")

        # 找出低谷时段
        low_hours = np.where(demand < np.percentile(demand, 25))[0]
        recommendations.append(f"低谷时段（{low_hours.tolist()}）可以减少人员安排")

        return recommendations