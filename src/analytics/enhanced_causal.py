import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dowhy import CausalModel
import networkx as nx
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型枚举"""
    RESTOCK = "补货"
    PROMOTION = "促销"
    PRICING = "调价"
    CUSTOMER_RECALL = "客户召回"
    CHANNEL_OPTIMIZATION = "渠道优化"
    PRODUCT_ADJUSTMENT = "商品调整"
    OPERATION_IMPROVEMENT = "运营改进"


@dataclass
class CausalAction:
    """因果动作定义"""
    action_type: ActionType
    action_name: str
    description: str
    target_metric: str
    expected_impact: float
    confidence_level: float
    priority: str
    implementation_steps: List[str]
    monitoring_metrics: List[str]


class EnhancedCausalInferenceEngine:
    """增强版因果推断引擎"""

    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.metric_action_mapping = self._init_metric_action_mapping()
        self.historical_effects = {}

    def _init_metric_action_mapping(self) -> Dict[str, List[CausalAction]]:
        """初始化指标-动作映射"""
        return {
            "sales_decline": [
                CausalAction(
                    action_type=ActionType.PROMOTION,
                    action_name="精准促销活动",
                    description="针对销量下滑商品推出限时优惠",
                    target_metric="daily_revenue",
                    expected_impact=0.15,
                    confidence_level=0.85,
                    priority="高",
                    implementation_steps=[
                        "识别销量下滑TOP 10商品",
                        "设计差异化折扣方案",
                        "选择目标客户群体",
                        "配置促销规则"
                    ],
                    monitoring_metrics=["conversion_rate", "aov", "customer_response_rate"]
                ),
                CausalAction(
                    action_type=ActionType.CUSTOMER_RECALL,
                    action_name="流失客户召回",
                    description="通过定向营销召回近期流失客户",
                    target_metric="customer_retention",
                    expected_impact=0.08,
                    confidence_level=0.75,
                    priority="中",
                    implementation_steps=[
                        "分析客户流失原因",
                        "设计个性化召回方案",
                        "多渠道触达（短信/邮件/App推送）",
                        "效果追踪与优化"
                    ],
                    monitoring_metrics=["recall_rate", "reactivation_rate", "ltv"]
                )
            ],
            "inventory_stockout": [
                CausalAction(
                    action_type=ActionType.RESTOCK,
                    action_name="紧急补货",
                    description="基于预测需求紧急补充库存",
                    target_metric="stockout_rate",
                    expected_impact=-0.90,
                    confidence_level=0.95,
                    priority="紧急",
                    implementation_steps=[
                        "计算安全库存水平",
                        "确定补货数量",
                        "选择快速供应商",
                        "加速配送流程"
                    ],
                    monitoring_metrics=["inventory_level", "lost_sales", "customer_satisfaction"]
                )
            ],
            "low_conversion": [
                CausalAction(
                    action_type=ActionType.PRICING,
                    action_name="动态定价优化",
                    description="基于需求弹性调整价格",
                    target_metric="conversion_rate",
                    expected_impact=0.12,
                    confidence_level=0.80,
                    priority="高",
                    implementation_steps=[
                        "分析价格敏感度",
                        "A/B测试不同价格点",
                        "实时监控转化率",
                        "动态调整策略"
                    ],
                    monitoring_metrics=["price_elasticity", "revenue_per_visitor", "competitive_position"]
                )
            ]
        }

    def analyze_metric_causality(
            self,
            data: pd.DataFrame,
            metric_name: str,
            metric_value: float,
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析单个指标的因果关系"""

        # 1. 检测指标异常程度
        anomaly_score = self._calculate_anomaly_score(data, metric_name, metric_value)

        # 2. 识别潜在原因
        root_causes = self._identify_root_causes(data, metric_name, context)

        # 3. 量化因果效应
        causal_effects = self._quantify_causal_effects(data, root_causes, metric_name)

        # 4. 生成行动建议
        recommended_actions = self._generate_action_recommendations(
            metric_name,
            anomaly_score,
            root_causes,
            causal_effects
        )

        # 5. 预测干预效果
        intervention_predictions = self._predict_intervention_effects(
            data,
            recommended_actions,
            metric_name
        )

        return {
            "metric": metric_name,
            "current_value": metric_value,
            "anomaly_score": anomaly_score,
            "root_causes": root_causes,
            "causal_effects": causal_effects,
            "recommended_actions": recommended_actions,
            "intervention_predictions": intervention_predictions,
            "confidence_level": self._calculate_overall_confidence(causal_effects)
        }

    def _identify_root_causes(
            self,
            data: pd.DataFrame,
            metric_name: str,
            context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别根本原因"""
        root_causes = []

        # 基于因果图的路径分析
        if metric_name == "daily_revenue":
            # 检查各个可能的原因
            checks = [
                ("traffic_decline", "visitor_count", -0.1),
                ("conversion_drop", "conversion_rate", -0.05),
                ("aov_decrease", "average_order_value", -0.1),
                ("stockout_impact", "stockout_count", 0.2),
                ("promotion_end", "active_promotions", -0.5)
            ]

            for cause_name, check_column, threshold in checks:
                if check_column in data.columns:
                    recent_change = self._calculate_recent_change(data, check_column)
                    if abs(recent_change) > abs(threshold):
                        root_causes.append({
                            "cause": cause_name,
                            "metric": check_column,
                            "change": recent_change,
                            "impact_score": abs(recent_change) / abs(threshold),
                            "confidence": min(0.95, 0.6 + 0.1 * len(data))
                        })

        # 按影响程度排序
        root_causes.sort(key=lambda x: x["impact_score"], reverse=True)
        return root_causes[:3]  # 返回TOP 3原因

    def _quantify_causal_effects(
            self,
            data: pd.DataFrame,
            root_causes: List[Dict[str, Any]],
            outcome_metric: str
    ) -> Dict[str, Dict[str, float]]:
        """量化因果效应"""
        effects = {}

        for cause in root_causes:
            cause_metric = cause["metric"]

            if cause_metric in data.columns and outcome_metric in data.columns:
                # 简化的因果效应计算
                # 实际应该使用 dowhy 或其他因果推断方法
                correlation = data[cause_metric].corr(data[outcome_metric])

                # 基于历史数据的效应估计
                if cause["cause"] in self.historical_effects:
                    historical_effect = self.historical_effects[cause["cause"]]
                else:
                    historical_effect = correlation * cause["change"]

                effects[cause["cause"]] = {
                    "estimated_effect": historical_effect,
                    "confidence_interval": (historical_effect * 0.8, historical_effect * 1.2),
                    "method": "historical_analysis",
                    "correlation": correlation
                }

        return effects

    def _generate_action_recommendations(
            self,
            metric_name: str,
            anomaly_score: float,
            root_causes: List[Dict[str, Any]],
            causal_effects: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """生成行动建议"""
        recommendations = []

        # 基于指标类型和根因选择动作
        if anomaly_score > 2.0:  # 显著异常
            if "sales_decline" in metric_name or any(c["cause"] == "traffic_decline" for c in root_causes):
                actions = self.metric_action_mapping.get("sales_decline", [])
                for action in actions:
                    recommendations.append({
                        "action": action,
                        "urgency": "immediate" if action.priority == "紧急" else "high",
                        "expected_roi": self._calculate_expected_roi(action, causal_effects),
                        "implementation_time": "1-2 days",
                        "risk_level": "low" if action.confidence_level > 0.8 else "medium"
                    })

        # 按预期ROI排序
        recommendations.sort(key=lambda x: x["expected_roi"], reverse=True)
        return recommendations[:3]

    def _predict_intervention_effects(
            self,
            data: pd.DataFrame,
            actions: List[Dict[str, Any]],
            target_metric: str
    ) -> Dict[str, Dict[str, Any]]:
        """预测干预效果"""
        predictions = {}

        for action_rec in actions:
            action = action_rec["action"]

            # 基于历史效果预测
            baseline = data[target_metric].mean() if target_metric in data.columns else 100

            predicted_impact = baseline * action.expected_impact

            predictions[action.action_name] = {
                "predicted_value": baseline + predicted_impact,
                "confidence_interval": (
                    baseline + predicted_impact * 0.7,
                    baseline + predicted_impact * 1.3
                ),
                "time_to_effect": "3-7 days",
                "success_probability": action.confidence_level
            }

        return predictions

    def run_comprehensive_causal_analysis(
            self,
            data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """运行综合因果分析"""
        results = {
            "timestamp": datetime.now(),
            "analyses": {},
            "global_recommendations": [],
            "causal_graph": None
        }

        # 分析各个关键指标
        key_metrics = [
            ("daily_revenue", "销售收入"),
            ("conversion_rate", "转化率"),
            ("inventory_turnover", "库存周转"),
            ("customer_retention", "客户留存")
        ]

        for metric_key, metric_name in key_metrics:
            if metric_key in data:
                metric_data = data[metric_key]
                current_value = metric_data.iloc[-1] if not metric_data.empty else 0

                analysis = self.analyze_metric_causality(
                    metric_data,
                    metric_key,
                    current_value,
                    {"historical_data": data}
                )

                results["analyses"][metric_key] = analysis

        # 生成全局建议
        results["global_recommendations"] = self._synthesize_global_recommendations(
            results["analyses"]
        )

        # 构建因果关系图
        results["causal_graph"] = self._build_causal_graph(data)

        return results

    def _calculate_anomaly_score(
            self,
            data: pd.DataFrame,
            metric_name: str,
            current_value: float
    ) -> float:
        """计算异常分数"""
        if metric_name not in data.columns:
            return 0.0

        historical = data[metric_name].dropna()
        if len(historical) < 2:
            return 0.0

        mean = historical.mean()
        std = historical.std()

        if std == 0:
            return 0.0

        z_score = abs(current_value - mean) / std
        return z_score

    def _calculate_recent_change(
            self,
            data: pd.DataFrame,
            column: str,
            days: int = 7
    ) -> float:
        """计算近期变化率"""
        if column not in data.columns:
            return 0.0

        if len(data) < days * 2:
            return 0.0

        recent = data[column].iloc[-days:].mean()
        previous = data[column].iloc[-days * 2:-days].mean()

        if previous == 0:
            return 0.0

        return (recent - previous) / previous

    def _calculate_expected_roi(
            self,
            action: CausalAction,
            causal_effects: Dict[str, Dict[str, float]]
    ) -> float:
        """计算预期投资回报率"""
        # 简化的ROI计算
        expected_benefit = action.expected_impact * 10000  # 假设基准收益
        estimated_cost = 1000  # 假设成本

        if estimated_cost == 0:
            return float('inf')

        roi = (expected_benefit - estimated_cost) / estimated_cost
        return roi

    def _calculate_overall_confidence(
            self,
            causal_effects: Dict[str, Dict[str, float]]
    ) -> float:
        """计算整体置信度"""
        if not causal_effects:
            return 0.5

        confidences = []
        for effect in causal_effects.values():
            if "correlation" in effect:
                confidences.append(abs(effect["correlation"]))

        return np.mean(confidences) if confidences else 0.5

    def _synthesize_global_recommendations(
            self,
            analyses: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """综合全局建议"""
        all_actions = []

        for metric, analysis in analyses.items():
            for rec in analysis.get("recommended_actions", []):
                all_actions.append({
                    "metric": metric,
                    "action": rec["action"],
                    "expected_roi": rec["expected_roi"],
                    "urgency": rec["urgency"]
                })

        # 去重和优先级排序
        unique_actions = {}
        for action in all_actions:
            key = action["action"].action_name
            if key not in unique_actions or action["expected_roi"] > unique_actions[key]["expected_roi"]:
                unique_actions[key] = action

        # 按ROI和紧急度排序
        sorted_actions = sorted(
            unique_actions.values(),
            key=lambda x: (x["urgency"] == "immediate", x["expected_roi"]),
            reverse=True
        )

        return sorted_actions[:5]

    def _build_causal_graph(
            self,
            data: Dict[str, pd.DataFrame]
    ) -> nx.DiGraph:
        """构建因果关系图"""
        G = nx.DiGraph()

        # 添加节点
        nodes = [
            ("Traffic", {"type": "source", "metrics": ["visitor_count", "page_views"]}),
            ("Conversion", {"type": "mediator", "metrics": ["conversion_rate", "cart_abandonment"]}),
            ("Revenue", {"type": "outcome", "metrics": ["daily_revenue", "aov"]}),
            ("Inventory", {"type": "mediator", "metrics": ["stockout_rate", "inventory_level"]}),
            ("Promotion", {"type": "treatment", "metrics": ["discount_rate", "promotion_count"]}),
            ("Customer", {"type": "mediator", "metrics": ["retention_rate", "churn_rate"]})
        ]

        for node, attrs in nodes:
            G.add_node(node, **attrs)

        # 添加边（因果关系）
        edges = [
            ("Traffic", "Conversion", {"effect": 0.7}),
            ("Conversion", "Revenue", {"effect": 0.9}),
            ("Inventory", "Conversion", {"effect": -0.3}),
            ("Promotion", "Conversion", {"effect": 0.4}),
            ("Promotion", "Revenue", {"effect": 0.3}),
            ("Customer", "Revenue", {"effect": 0.6}),
            ("Traffic", "Revenue", {"effect": 0.2})
        ]

        G.add_edges_from([(u, v, d) for u, v, d in edges])

        self.causal_graph = G
        return G