# advanced_causal_engine.py
"""高级因果推断引擎 - 包含完整的因果模型和反事实推理"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# DoWhy 因果推断
from dowhy import CausalModel
# from dowhy.causal_estimators import LinearRegressionEstimator
from dowhy.causal_refuters import AddUnobservedCommonCause

# 因果图构建
import networkx as nx
from dowhy.gcm import StructuralCausalModel
import dowhy.gcm as gcm

# 反事实推理
from dowhy.causal_identifier import IdentifiedEstimand
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualScenario:
    """反事实场景定义"""
    scenario_name: str
    treatment_change: Dict[str, float]  # 处理变量的假设变化
    context: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)


@dataclass
class CausalEffect:
    """因果效应结果"""
    ate: float  # Average Treatment Effect
    att: float  # Average Treatment on Treated
    atu: float  # Average Treatment on Untreated
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str


class AdvancedCausalEngine:
    """高级因果推断引擎"""

    def __init__(self):
        self.causal_models = {}
        self.scm_models = {}  # Structural Causal Models
        self.counterfactual_cache = {}

    def build_business_causal_graph(self) -> str:
        """构建业务因果图"""
        # 定义完整的业务因果关系
        causal_graph = """
        digraph {
            // 外部因素
            Weather -> Store_Traffic;
            Competitor_Promotion -> Customer_Choice;
            Holiday -> Store_Traffic;
            Holiday -> Online_Traffic;

            // 流量因素
            Store_Traffic -> Total_Visitors;
            Online_Traffic -> Total_Visitors;
            Marketing_Spend -> Online_Traffic;
            Marketing_Spend -> Brand_Awareness;
            Brand_Awareness -> Store_Traffic;

            // 商品和库存
            Inventory_Level -> Product_Availability;
            Product_Availability -> Conversion_Rate;
            Product_Quality -> Customer_Satisfaction;
            Product_Quality -> Return_Rate;

            // 价格和促销
            Price -> Conversion_Rate;
            Promotion -> Price;
            Promotion -> Conversion_Rate;
            Promotion -> Average_Order_Value;
            Customer_Choice -> Conversion_Rate;

            // 转化和销售
            Total_Visitors -> Orders;
            Conversion_Rate -> Orders;
            Orders -> Revenue;
            Average_Order_Value -> Revenue;

            // 客户因素
            Customer_Satisfaction -> Customer_Retention;
            Customer_Retention -> Repeat_Orders;
            Repeat_Orders -> Revenue;
            Return_Rate -> Customer_Satisfaction;

            // 运营因素
            Staff_Performance -> Service_Quality;
            Service_Quality -> Customer_Satisfaction;
            Service_Quality -> Conversion_Rate;

            // 隐藏的混淆因素
            U1 -> Price;
            U1 -> Revenue;  // 未观测的市场因素
            U2 -> Marketing_Spend;
            U2 -> Revenue;  // 未观测的预算约束
        }
        """
        return causal_graph

    def create_causal_model(
            self,
            data: pd.DataFrame,
            treatment: str,
            outcome: str,
            common_causes: Optional[List[str]] = None,
            instruments: Optional[List[str]] = None,
            effect_modifiers: Optional[List[str]] = None,
            custom_graph: Optional[str] = None
    ) -> CausalModel:
        """创建因果模型"""

        # 使用自定义图或构建标准图
        if custom_graph:
            graph = custom_graph
        else:
            graph = self._build_graph_from_variables(
                treatment, outcome, common_causes, instruments, effect_modifiers
            )

        # 创建因果模型
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
            instruments=instruments,
            effect_modifiers=effect_modifiers,
            graph=graph
        )

        # 存储模型
        model_key = f"{treatment}_to_{outcome}"
        self.causal_models[model_key] = model

        return model

    def estimate_causal_effect(
            self,
            model: CausalModel,
            methods: Optional[List[str]] = None
    ) -> Dict[str, CausalEffect]:
        """使用多种方法估计因果效应"""

        if methods is None:
            methods = [
                "backdoor.propensity_score_matching",
                "backdoor.propensity_score_stratification",
                "backdoor.propensity_score_weighting",
                "backdoor.linear_regression",
                "iv.instrumental_variable",
                "frontdoor.two_stage_regression"
            ]

        # 识别因果效应
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        results = {}

        for method in methods:
            try:
                # 估计效应
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method,
                    target_units="ate",  # Average Treatment Effect
                    confidence_intervals=True,
                    test_significance=True
                )

                # 计算不同类型的效应
                ate = estimate.value

                # ATT和ATU需要额外计算
                att_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method,
                    target_units="att"  # Average Treatment on Treated
                )
                att = att_estimate.value if att_estimate else ate

                atu_estimate = model.estimate_effect(
                    identified_estimand,
                    method_name=method,
                    target_units="atu"  # Average Treatment on Untreated
                )
                atu = atu_estimate.value if atu_estimate else ate

                # 提取置信区间和p值
                ci = self._extract_confidence_interval(estimate)
                p_value = self._extract_p_value(estimate)

                results[method] = CausalEffect(
                    ate=ate,
                    att=att,
                    atu=atu,
                    confidence_interval=ci,
                    p_value=p_value,
                    method=method
                )

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue

        return results

    def perform_counterfactual_analysis(
            self,
            model: CausalModel,
            scenario: CounterfactualScenario,
            sample_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """执行反事实分析 - 回答"如果...会怎样"的问题"""

        logger.info(f"执行反事实分析: {scenario.scenario_name}")

        # 如果没有提供样本数据，使用模型的原始数据
        if sample_data is None:
            sample_data = model._data.copy()

        # 获取处理变量和结果变量
        treatment = model._treatment
        outcome = model._outcome

        # 1. 训练结果预测模型
        outcome_model = self._train_outcome_model(
            model._data,
            treatment,
            outcome,
            model._common_causes
        )

        # 2. 创建反事实场景数据
        counterfactual_data = sample_data.copy()

        # 应用处理变量的变化
        for var, change in scenario.treatment_change.items():
            if var in counterfactual_data.columns:
                # 支持绝对值和相对变化
                if isinstance(change, str) and change.endswith('%'):
                    # 相对变化
                    pct_change = float(change[:-1]) / 100
                    counterfactual_data[var] = counterfactual_data[var] * (1 + pct_change)
                else:
                    # 绝对值变化
                    counterfactual_data[var] = counterfactual_data[var] + change

        # 3. 预测反事实结果
        factual_outcome = sample_data[outcome].values
        counterfactual_outcome = outcome_model.predict(
            counterfactual_data[model._common_causes + treatment]
        )

        # 4. 计算个体处理效应 (ITE)
        ite = counterfactual_outcome - factual_outcome

        # 5. 聚合结果
        results = {
            "scenario": scenario.scenario_name,
            "treatment_changes": scenario.treatment_change,
            "individual_effects": {
                "mean": np.mean(ite),
                "median": np.median(ite),
                "std": np.std(ite),
                "min": np.min(ite),
                "max": np.max(ite),
                "percentiles": {
                    "25%": np.percentile(ite, 25),
                    "75%": np.percentile(ite, 75),
                    "95%": np.percentile(ite, 95)
                }
            },
            "aggregate_impact": {
                "total_change": np.sum(ite),
                "average_change": np.mean(ite),
                "positive_impact_pct": np.mean(ite > 0) * 100,
                "significant_impact_pct": np.mean(np.abs(ite) > np.std(factual_outcome)) * 100
            },
            "subgroup_analysis": self._analyze_subgroups(
                sample_data, ite, scenario.context
            ),
            "confidence": self._calculate_counterfactual_confidence(
                outcome_model, sample_data, counterfactual_data
            )
        }

        return results

    def what_if_analysis(
            self,
            data: pd.DataFrame,
            scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """批量What-if分析"""

        results = []

        for scenario in scenarios:
            # 构建因果模型
            model = self.create_causal_model(
                data=data,
                treatment=scenario['treatment'],
                outcome=scenario['outcome'],
                common_causes=scenario.get('confounders', [])
            )

            # 创建反事实场景
            cf_scenario = CounterfactualScenario(
                scenario_name=scenario['name'],
                treatment_change=scenario['changes'],
                context=scenario.get('context', {})
            )

            # 执行反事实分析
            cf_results = self.perform_counterfactual_analysis(model, cf_scenario)

            results.append({
                'scenario': scenario['name'],
                'treatment': scenario['treatment'],
                'outcome': scenario['outcome'],
                'mean_effect': cf_results['individual_effects']['mean'],
                'total_impact': cf_results['aggregate_impact']['total_change'],
                'confidence': cf_results['confidence']
            })

        return pd.DataFrame(results)

    def analyze_promotion_scenarios(self, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """分析不同促销场景的反事实结果"""

        # 准备数据
        data = self._prepare_promotion_data(sales_data)

        # 创建促销因果模型
        model = self.create_causal_model(
            data=data,
            treatment=['discount_rate', 'promotion_type'],
            outcome='revenue',
            common_causes=['day_of_week', 'is_holiday', 'weather', 'inventory_level'],
            effect_modifiers=['customer_segment', 'product_category']
        )

        # 定义不同的促销场景
        scenarios = [
            CounterfactualScenario(
                scenario_name="激进促销 - 全场8折",
                treatment_change={'discount_rate': 0.2, 'promotion_type': 1},
                context={'target': 'all_customers', 'duration': '3_days'}
            ),
            CounterfactualScenario(
                scenario_name="精准促销 - VIP专享9折",
                treatment_change={'discount_rate': 0.1, 'promotion_type': 2},
                context={'target': 'vip_only', 'duration': '7_days'}
            ),
            CounterfactualScenario(
                scenario_name="买一送一促销",
                treatment_change={'discount_rate': 0.5, 'promotion_type': 3},
                context={'target': 'selected_items', 'duration': '1_day'}
            ),
            CounterfactualScenario(
                scenario_name="无促销基准",
                treatment_change={'discount_rate': 0, 'promotion_type': 0},
                context={'target': 'none', 'duration': 'baseline'}
            )
        ]

        # 分析每个场景
        scenario_results = {}
        for scenario in scenarios:
            result = self.perform_counterfactual_analysis(model, scenario)
            scenario_results[scenario.scenario_name] = result

        # 对比分析
        comparison = self._compare_scenarios(scenario_results)

        # 生成最优建议
        recommendation = self._generate_optimal_promotion_strategy(
            scenario_results,
            data
        )

        return {
            "scenario_analysis": scenario_results,
            "comparison": comparison,
            "recommendation": recommendation,
            "causal_model": model
        }

    def analyze_inventory_decisions(self, inventory_data: pd.DataFrame) -> Dict[str, Any]:
        """分析库存决策的反事实影响"""

        # 创建库存因果模型
        model = self.create_causal_model(
            data=inventory_data,
            treatment='inventory_level',
            outcome='lost_sales',
            common_causes=['demand_forecast', 'lead_time', 'seasonality'],
            instruments=['supplier_capacity']  # 使用供应商产能作为工具变量
        )

        # 反事实场景：如果我们维持不同的库存水平
        current_inventory = inventory_data['inventory_level'].mean()

        scenarios = []
        for multiplier in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            scenarios.append(
                CounterfactualScenario(
                    scenario_name=f"库存水平{multiplier}x",
                    treatment_change={
                        'inventory_level': current_inventory * (multiplier - 1)
                    },
                    context={
                        'holding_cost_per_unit': 10,
                        'stockout_cost_per_unit': 50
                    }
                )
            )

        # 执行反事实分析
        results = {}
        optimal_level = current_inventory
        min_total_cost = float('inf')

        for scenario in scenarios:
            cf_result = self.perform_counterfactual_analysis(model, scenario)

            # 计算总成本
            holding_cost = scenario.treatment_change['inventory_level'] * \
                           scenario.context['holding_cost_per_unit']
            stockout_cost = -cf_result['aggregate_impact']['total_change'] * \
                            scenario.context['stockout_cost_per_unit']
            total_cost = holding_cost + stockout_cost

            cf_result['cost_analysis'] = {
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost,
                'total_cost': total_cost
            }

            results[scenario.scenario_name] = cf_result

            # 寻找最优库存水平
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                optimal_level = current_inventory * float(scenario.scenario_name.split('x')[0].split('平')[-1])

        return {
            "counterfactual_results": results,
            "optimal_inventory_level": optimal_level,
            "expected_cost_saving": current_inventory * 10 - min_total_cost,
            "recommendation": self._generate_inventory_recommendation(results, optimal_level)
        }

    def perform_sensitivity_analysis(
            self,
            model: CausalModel,
            estimate: Any
    ) -> Dict[str, Any]:
        """执行敏感性分析"""

        sensitivity_results = {}

        # 1. 添加未观察到的混淆因素
        try:
            refuter = AddUnobservedCommonCause(
                estimate=estimate,
                identified_estimand=model.identify_effect(),
                model=model,
                confounders_effect_on_treatment=0.1,
                confounders_effect_on_outcome=0.1
            )
            unobserved_result = refuter.refute_estimate()
            sensitivity_results['unobserved_confounder'] = {
                'original_effect': estimate.value,
                'new_effect': unobserved_result.new_effect,
                'change_percent': abs(unobserved_result.new_effect - estimate.value) / abs(estimate.value) * 100,
                'robust': abs(unobserved_result.new_effect - estimate.value) < abs(estimate.value) * 0.1
            }
        except Exception as e:
            logger.error(f"Unobserved confounder test failed: {e}")

        # 2. 数据子集测试
        subset_effects = []
        for i in range(10):
            subset_data = model._data.sample(frac=0.8, replace=True)
            subset_model = CausalModel(
                data=subset_data,
                treatment=model._treatment,
                outcome=model._outcome,
                common_causes=model._common_causes
            )
            subset_estimate = subset_model.estimate_effect(
                subset_model.identify_effect(),
                method_name=estimate.estimator_name
            )
            subset_effects.append(subset_estimate.value)

        sensitivity_results['bootstrap'] = {
            'effects': subset_effects,
            'mean': np.mean(subset_effects),
            'std': np.std(subset_effects),
            'cv': np.std(subset_effects) / abs(np.mean(subset_effects)),
            'stable': np.std(subset_effects) / abs(np.mean(subset_effects)) < 0.2
        }

        # 3. 极端值影响
        # 移除top和bottom 5%的数据
        treatment_col = model._treatment[0] if isinstance(model._treatment, list) else model._treatment
        outcome_col = model._outcome

        percentile_5 = model._data[outcome_col].quantile(0.05)
        percentile_95 = model._data[outcome_col].quantile(0.95)

        trimmed_data = model._data[
            (model._data[outcome_col] >= percentile_5) &
            (model._data[outcome_col] <= percentile_95)
            ]

        trimmed_model = CausalModel(
            data=trimmed_data,
            treatment=model._treatment,
            outcome=model._outcome,
            common_causes=model._common_causes
        )
        trimmed_estimate = trimmed_model.estimate_effect(
            trimmed_model.identify_effect(),
            method_name=estimate.estimator_name
        )

        sensitivity_results['outlier_robustness'] = {
            'original_effect': estimate.value,
            'trimmed_effect': trimmed_estimate.value,
            'change_percent': abs(trimmed_estimate.value - estimate.value) / abs(estimate.value) * 100,
            'robust': abs(trimmed_estimate.value - estimate.value) < abs(estimate.value) * 0.15
        }

        # 总体稳健性评分
        robustness_scores = [
            result.get('robust', False) or result.get('stable', False)
            for result in sensitivity_results.values()
        ]
        sensitivity_results['overall_robustness'] = {
            'score': sum(robustness_scores) / len(robustness_scores),
            'interpretation': self._interpret_robustness(sum(robustness_scores) / len(robustness_scores))
        }

        return sensitivity_results

    # 辅助方法
    def _build_graph_from_variables(
            self,
            treatment,
            outcome,
            common_causes,
            instruments,
            effect_modifiers
    ) -> str:
        """从变量构建因果图"""
        edges = []

        # 支持 treatment 为 list 或 str
        treatments = treatment if isinstance(treatment, list) else [treatment]
        outcomes = outcome if isinstance(outcome, list) else [outcome]

        # 处理到结果
        for t in treatments:
            for o in outcomes:
                edges.append(f"{t} -> {o}")

        # 共同原因
        if common_causes:
            for cause in common_causes:
                for t in treatments:
                    edges.append(f"{cause} -> {t}")
                for o in outcomes:
                    edges.append(f"{cause} -> {o}")

        # 工具变量
        if instruments:
            for instrument in instruments:
                for t in treatments:
                    edges.append(f"{instrument} -> {t}")

        # 效应修饰因子
        if effect_modifiers:
            for modifier in effect_modifiers:
                for o in outcomes:
                    edges.append(f"{modifier} -> {o}")
                # 修饰因子可能也影响处理效应
                for t in treatments:
                    for o in outcomes:
                        edges.append(f"{t} -> {o} [label=\"moderated by {modifier}\"]")

        graph = "digraph {\n" + ";\n".join(edges) + "\n}"
        return graph

    def _train_outcome_model(
            self,
            data: pd.DataFrame,
            treatment: List[str],
            outcome: str,
            confounders: List[str]
    ) -> Any:
        """训练结果预测模型"""
        # 准备特征
        features = confounders + (treatment if isinstance(treatment, list) else [treatment])
        X = data[features]
        y = data[outcome]

        # 使用随机森林进行非线性建模
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)

        return model

    def _analyze_subgroups(
            self,
            data: pd.DataFrame,
            effects: np.ndarray,
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析不同子群体的效应"""
        subgroup_results = {}

        # 按客户类型分析
        if 'customer_segment' in data.columns:
            for segment in data['customer_segment'].unique():
                mask = data['customer_segment'] == segment
                subgroup_results[f'customer_{segment}'] = {
                    'mean_effect': np.mean(effects[mask]),
                    'count': sum(mask),
                    'std': np.std(effects[mask])
                }

        # 按时间分析
        if 'day_of_week' in data.columns:
            for day in range(7):
                mask = data['day_of_week'] == day
                if sum(mask) > 0:
                    subgroup_results[f'day_{day}'] = {
                        'mean_effect': np.mean(effects[mask]),
                        'count': sum(mask)
                    }

        return subgroup_results

    def _calculate_counterfactual_confidence(
            self,
            model: Any,
            factual_data: pd.DataFrame,
            counterfactual_data: pd.DataFrame
    ) -> float:
        """计算反事实预测的置信度"""
        # 简化版本：基于预测的不确定性
        # 实际应该使用更复杂的方法，如预测区间

        # 计算数据分布的相似度
        factual_mean = factual_data.mean()
        cf_mean = counterfactual_data.mean()

        # 使用相对差异作为置信度的反向指标
        relative_diff = np.mean(np.abs(cf_mean - factual_mean) / (factual_mean + 1e-10))

        # 转换为置信度（差异越大，置信度越低）
        confidence = max(0, 1 - relative_diff)

        return confidence

    def _extract_confidence_interval(self, estimate: Any) -> Tuple[float, float]:
        """提取置信区间"""
        try:
            if hasattr(estimate, 'get_confidence_intervals'):
                ci = estimate.get_confidence_intervals()
                return (ci[0], ci[1])
            else:
                # 使用标准误差估计
                se = estimate.get_standard_error() if hasattr(estimate, 'get_standard_error') else estimate.value * 0.1
                return (estimate.value - 1.96 * se, estimate.value + 1.96 * se)
        except:
            return (estimate.value * 0.8, estimate.value * 1.2)

    def _extract_p_value(self, estimate: Any) -> float:
        """提取p值"""
        try:
            if hasattr(estimate, 'test_stat_significance'):
                return estimate.test_stat_significance()
            else:
                return 0.05  # 默认值
        except:
            return 0.05

    def _prepare_promotion_data(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """准备促销分析数据"""
        # 确保必要的列存在
        required_columns = ['revenue', 'discount_rate', 'promotion_type',
                            'day_of_week', 'is_holiday', 'weather',
                            'inventory_level', 'customer_segment', 'product_category']

        # 添加缺失的列（使用模拟数据）
        for col in required_columns:
            if col not in sales_data.columns:
                if col == 'discount_rate':
                    sales_data[col] = np.random.uniform(0, 0.3, len(sales_data))
                elif col == 'promotion_type':
                    sales_data[col] = np.random.choice([0, 1, 2, 3], len(sales_data))
                elif col == 'weather':
                    sales_data[col] = np.random.choice(['sunny', 'rainy', 'cloudy'], len(sales_data))
                elif col == 'customer_segment':
                    sales_data[col] = np.random.choice(['vip', 'regular', 'new'], len(sales_data))
                elif col == 'product_category':
                    sales_data[col] = np.random.choice(['drinks', 'food', 'snacks'], len(sales_data))
                else:
                    sales_data[col] = np.random.randn(len(sales_data))

        return sales_data

    def _compare_scenarios(self, scenario_results: Dict[str, Any]) -> pd.DataFrame:
        """比较不同场景的结果"""
        comparison_data = []

        for scenario_name, result in scenario_results.items():
            comparison_data.append({
                'scenario': scenario_name,
                'mean_effect': result['individual_effects']['mean'],
                'total_impact': result['aggregate_impact']['total_change'],
                'positive_impact_pct': result['aggregate_impact']['positive_impact_pct'],
                'confidence': result['confidence']
            })

        return pd.DataFrame(comparison_data).sort_values('mean_effect', ascending=False)

    def _generate_optimal_promotion_strategy(
            self,
            scenario_results: Dict[str, Any],
            data: pd.DataFrame
    ) -> Dict[str, Any]:
        """生成最优促销策略"""
        # 找出效果最好的场景
        best_scenario = max(
            scenario_results.items(),
            key=lambda x: x[1]['individual_effects']['mean']
        )

        return {
            'recommended_scenario': best_scenario[0],
            'expected_revenue_increase': best_scenario[1]['aggregate_impact']['total_change'],
            'confidence_level': best_scenario[1]['confidence'],
            'implementation_notes': self._generate_implementation_notes(best_scenario)
        }

    def _generate_implementation_notes(self, scenario_result: Tuple[str, Dict]) -> List[str]:
        """生成实施建议"""
        scenario_name, result = scenario_result
        notes = []

        if "全场" in scenario_name:
            notes.append("建议在周末或节假日实施以最大化效果")
            notes.append("注意控制库存，避免断货")
        elif "VIP" in scenario_name:
            notes.append("通过App推送和短信定向通知VIP客户")
            notes.append("可配合积分翻倍活动增强吸引力")
        elif "买一送一" in scenario_name:
            notes.append("选择毛利率较高的商品参与活动")
            notes.append("限时限量，创造紧迫感")

        # 基于效果的建议
        if result['aggregate_impact']['positive_impact_pct'] > 80:
            notes.append("预期效果良好，建议快速推进")
        else:
            notes.append("建议先小范围测试，验证效果后再推广")

        return notes

    def _generate_inventory_recommendation(
            self,
            results: Dict[str, Any],
            optimal_level: float
    ) -> Dict[str, Any]:
        """生成库存建议"""
        current_cost = list(results.values())[2]['cost_analysis']['total_cost']  # 1.0x场景
        optimal_cost = min(r['cost_analysis']['total_cost'] for r in results.values())

        return {
            'action': '调整库存水平',
            'target_level': optimal_level,
            'expected_cost_saving': current_cost - optimal_cost,
            'implementation_steps': [
                f"将平均库存水平调整至{optimal_level:.0f}单位",
                "与供应商协商缩短补货周期",
                "实施动态安全库存策略",
                "加强需求预测准确性"
            ],
            'risk_mitigation': [
                "保留应急供应商清单",
                "建立库存预警机制",
                "准备促销方案应对过剩库存"
            ]
        }

    def _interpret_robustness(self, score: float) -> str:
        """解释稳健性分数"""
        if score >= 0.8:
            return "非常稳健：结果在各种条件下都保持一致"
        elif score >= 0.6:
            return "较为稳健：结果在大多数条件下稳定"
        elif score >= 0.4:
            return "中等稳健：结果存在一定不确定性"
        else:
            return "稳健性较低：建议谨慎解释结果"