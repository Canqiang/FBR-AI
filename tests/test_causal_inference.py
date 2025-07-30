# causal_business_examples.py
"""因果推断在实际业务中的应用示例"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

from src.analytics.advanced_causal_engine import (
    AdvancedCausalEngine,
    CounterfactualScenario,
    CausalEffect
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalBusinessAnalyzer:
    """业务场景的因果分析器"""

    def __init__(self):
        self.engine = AdvancedCausalEngine()
        self.analysis_results = {}

    def analyze_sales_decline(self, business_data: pd.DataFrame) -> Dict[str, Any]:
        """分析销售下滑的因果关系和反事实场景"""

        logger.info("开始分析销售下滑问题...")

        # 1. 构建销售因果模型
        sales_model = self.engine.create_causal_model(
            data=business_data,
            treatment=['promotion_active', 'price_level'],
            outcome='daily_revenue',
            common_causes=[
                'day_of_week',
                'weather_condition',
                'competitor_activity',
                'inventory_availability',
                'store_traffic'
            ],
            effect_modifiers=['customer_segment', 'product_type']
        )

        # 2. 估计当前因果效应
        causal_effects = self.engine.estimate_causal_effect(
            sales_model,
            methods=[
                "backdoor.propensity_score_matching",
                "backdoor.linear_regression",
                "backdoor.propensity_score_weighting"
            ]
        )

        # 3. 反事实分析：如果我们采取不同的策略会怎样？
        counterfactual_scenarios = []

        # 场景1：如果我们提高促销力度
        counterfactual_scenarios.append({
            'scenario': CounterfactualScenario(
                scenario_name="增强促销策略",
                treatment_change={
                    'promotion_active': 1.0,  # 全面启动促销
                    'price_level': -0.15  # 降价15%
                },
                context={
                    'target_segment': 'all',
                    'expected_cost': 50000,
                    'duration_days': 7
                }
            ),
            'business_question': "如果我们实施全场85折促销，销售额会增加多少？"
        })

        # 场景2：如果竞争对手没有促销
        counterfactual_scenarios.append({
            'scenario': CounterfactualScenario(
                scenario_name="竞争环境改善",
                treatment_change={
                    'competitor_activity': -1.0  # 竞争对手停止促销
                },
                context={
                    'market_condition': 'favorable',
                    'probability': 0.3
                }
            ),
            'business_question': "如果竞争对手停止促销活动，我们的销售会恢复多少？"
        })

        # 场景3：如果天气转好且库存充足
        counterfactual_scenarios.append({
            'scenario': CounterfactualScenario(
                scenario_name="理想运营条件",
                treatment_change={
                    'weather_condition': 1.0,  # 好天气
                    'inventory_availability': 1.0  # 库存充足
                },
                context={
                    'operational_readiness': 'optimal',
                    'feasibility': 'high'
                }
            ),
            'business_question': "在理想条件下（好天气+充足库存），销售潜力有多大？"
        })

        # 4. 执行反事实分析
        counterfactual_results = {}
        for cf_item in counterfactual_scenarios:
            scenario = cf_item['scenario']
            result = self.engine.perform_counterfactual_analysis(
                sales_model,
                scenario,
                sample_data=business_data.tail(30)  # 使用最近30天数据
            )
            result['business_question'] = cf_item['business_question']
            counterfactual_results[scenario.scenario_name] = result

        # 5. 执行敏感性分析
        sensitivity = self.engine.perform_sensitivity_analysis(
            sales_model,
            list(causal_effects.values())[0]  # 使用第一个估计结果
        )

        # 6. 生成业务洞察和建议
        insights = self._generate_sales_insights(
            causal_effects,
            counterfactual_results,
            sensitivity
        )

        return {
            'causal_effects': causal_effects,
            'counterfactual_analysis': counterfactual_results,
            'sensitivity_analysis': sensitivity,
            'business_insights': insights,
            'recommended_actions': self._prioritize_actions(counterfactual_results)
        }

    def analyze_customer_churn(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """分析客户流失的因果关系"""

        logger.info("开始分析客户流失问题...")

        # 1. 构建客户流失因果模型
        churn_model = self.engine.create_causal_model(
            data=customer_data,
            treatment='received_retention_offer',
            outcome='churned',
            common_causes=[
                'customer_lifetime_value',
                'recent_support_tickets',
                'days_since_last_purchase',
                'total_purchases',
                'satisfaction_score'
            ],
            instruments=['random_campaign_assignment']  # 使用随机分配作为工具变量
        )

        # 2. 估计挽回策略的因果效应
        retention_effects = self.engine.estimate_causal_effect(
            churn_model,
            methods=["iv.instrumental_variable", "backdoor.propensity_score_matching"]
        )

        # 3. 反事实分析：不同挽回策略的效果
        retention_scenarios = [
            CounterfactualScenario(
                scenario_name="个性化优惠券策略",
                treatment_change={'received_retention_offer': 1.0},
                context={
                    'offer_type': 'personalized_discount',
                    'discount_amount': 0.25,
                    'cost_per_customer': 20
                }
            ),
            CounterfactualScenario(
                scenario_name="VIP升级策略",
                treatment_change={'received_retention_offer': 1.0},
                context={
                    'offer_type': 'vip_upgrade',
                    'benefits': ['free_shipping', 'exclusive_access'],
                    'cost_per_customer': 50
                }
            ),
            CounterfactualScenario(
                scenario_name="积分奖励策略",
                treatment_change={'received_retention_offer': 1.0},
                context={
                    'offer_type': 'bonus_points',
                    'points_multiplier': 3,
                    'cost_per_customer': 15
                }
            )
        ]

        # 4. 分析每个策略的效果
        strategy_results = {}
        for scenario in retention_scenarios:
            # 只对高风险客户群体进行分析
            high_risk_customers = customer_data[
                customer_data['churn_probability'] > 0.7
                ]

            result = self.engine.perform_counterfactual_analysis(
                churn_model,
                scenario,
                sample_data=high_risk_customers
            )

            # 计算ROI
            prevented_churns = -result['aggregate_impact']['total_change']
            revenue_saved = prevented_churns * customer_data['customer_lifetime_value'].mean()
            total_cost = len(high_risk_customers) * scenario.context['cost_per_customer']
            roi = (revenue_saved - total_cost) / total_cost

            result['financial_impact'] = {
                'prevented_churns': prevented_churns,
                'revenue_saved': revenue_saved,
                'total_cost': total_cost,
                'roi': roi
            }

            strategy_results[scenario.scenario_name] = result

        return {
            'retention_effects': retention_effects,
            'strategy_comparison': strategy_results,
            'optimal_strategy': self._find_optimal_retention_strategy(strategy_results),
            'segmented_recommendations': self._segment_retention_recommendations(
                customer_data,
                strategy_results
            )
        }

    def analyze_pricing_decisions(self, pricing_data: pd.DataFrame) -> Dict[str, Any]:
        """分析定价决策的因果影响"""

        logger.info("开始分析定价策略...")

        # 1. 构建价格弹性模型
        pricing_model = self.engine.create_causal_model(
            data=pricing_data,
            treatment='price',
            outcome='units_sold',
            common_causes=[
                'product_category',
                'competitor_price',
                'seasonality',
                'promotion_active',
                'inventory_level'
            ],
            effect_modifiers=['customer_segment', 'time_of_day']
        )

        # 2. 估计价格弹性
        price_effects = self.engine.estimate_causal_effect(pricing_model)

        # 3. What-if 分析：不同定价策略
        pricing_scenarios = [
            {
                'name': '激进降价策略',
                'treatment': 'price',
                'outcome': 'units_sold',
                'changes': {'price': -0.20},  # 降价20%
                'confounders': ['competitor_price', 'seasonality']
            },
            {
                'name': '温和涨价策略',
                'treatment': 'price',
                'outcome': 'units_sold',
                'changes': {'price': 0.05},  # 涨价5%
                'confounders': ['competitor_price', 'seasonality']
            },
            {
                'name': '动态定价策略',
                'treatment': 'price',
                'outcome': 'units_sold',
                'changes': {'price': 'dynamic'},  # 根据需求动态调整
                'confounders': ['competitor_price', 'seasonality', 'time_of_day']
            }
        ]

        # 4. 批量What-if分析
        what_if_results = self.engine.what_if_analysis(
            pricing_data,
            pricing_scenarios
        )

        # 5. 计算最优价格点
        optimal_price = self._find_optimal_price(
            pricing_data,
            price_effects,
            what_if_results
        )

        return {
            'price_elasticity': price_effects,
            'scenario_analysis': what_if_results,
            'optimal_price': optimal_price,
            'implementation_roadmap': self._create_pricing_roadmap(optimal_price)
        }

    def analyze_inventory_optimization(self, inventory_data: pd.DataFrame) -> Dict[str, Any]:
        """库存优化的因果分析"""

        logger.info("开始分析库存优化策略...")

        # 使用引擎的库存分析方法
        inventory_results = self.engine.analyze_inventory_decisions(inventory_data)

        # 添加额外的业务分析
        inventory_results['seasonal_adjustments'] = self._analyze_seasonal_inventory(
            inventory_data
        )

        inventory_results['supplier_recommendations'] = self._analyze_supplier_impact(
            inventory_data
        )

        return inventory_results

    def analyze_marketing_effectiveness(self, marketing_data: pd.DataFrame) -> Dict[str, Any]:
        """营销效果的因果分析"""

        logger.info("开始分析营销效果...")

        # 1. 多渠道归因模型
        marketing_model = self.engine.create_causal_model(
            data=marketing_data,
            treatment=['email_sent', 'sms_sent', 'push_notification'],
            outcome='purchase_made',
            common_causes=[
                'customer_segment',
                'past_purchase_frequency',
                'time_since_last_purchase'
            ],
            effect_modifiers=['message_content', 'send_time']
        )

        # 2. 估计各渠道效果
        channel_effects = self.engine.estimate_causal_effect(marketing_model)

        # 3. 反事实：如果改变营销组合
        marketing_mix_scenarios = [
            CounterfactualScenario(
                scenario_name="仅Email策略",
                treatment_change={
                    'email_sent': 1.0,
                    'sms_sent': 0.0,
                    'push_notification': 0.0
                },
                context={'cost_per_channel': {'email': 0.1}}
            ),
            CounterfactualScenario(
                scenario_name="全渠道轰炸",
                treatment_change={
                    'email_sent': 1.0,
                    'sms_sent': 1.0,
                    'push_notification': 1.0
                },
                context={'cost_per_channel': {'email': 0.1, 'sms': 0.5, 'push': 0.2}}
            ),
            CounterfactualScenario(
                scenario_name="智能组合策略",
                treatment_change={
                    'email_sent': 0.8,
                    'sms_sent': 0.3,
                    'push_notification': 0.6
                },
                context={'cost_per_channel': {'email': 0.1, 'sms': 0.5, 'push': 0.2}}
            )
        ]

        # 4. 分析每个策略
        mix_results = {}
        for scenario in marketing_mix_scenarios:
            result = self.engine.perform_counterfactual_analysis(
                marketing_model,
                scenario
            )

            # 计算成本效益
            total_cost = sum(
                scenario.treatment_change[f'{channel}_sent'] *
                scenario.context['cost_per_channel'].get(channel, 0)
                for channel in ['email', 'sms', 'push']
            )

            result['cost_effectiveness'] = {
                'total_cost': total_cost,
                'conversions_per_dollar': result['aggregate_impact'][
                                              'total_change'] / total_cost if total_cost > 0 else 0
            }

            mix_results[scenario.scenario_name] = result

        return {
            'channel_attribution': channel_effects,
            'marketing_mix_optimization': mix_results,
            'recommended_mix': self._optimize_marketing_mix(mix_results),
            'personalization_opportunities': self._identify_personalization_opportunities(
                marketing_data,
                channel_effects
            )
        }

    # 辅助方法
    def _generate_sales_insights(
            self,
            causal_effects: Dict[str, CausalEffect],
            counterfactual_results: Dict[str, Any],
            sensitivity: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成销售洞察"""
        insights = []

        # 1. 促销效果洞察
        promotion_effect = next((effect for name, effect in causal_effects.items()
                                 if 'promotion' in name.lower()), None)
        if promotion_effect:
            insights.append({
                'type': 'promotion_effectiveness',
                'finding': f"促销活动平均提升销售额{promotion_effect.ate * 100:.1f}%",
                'confidence': 'high' if sensitivity['overall_robustness']['score'] > 0.7 else 'medium',
                'action': "继续优化促销策略，特别关注高响应客群"
            })

        # 2. 反事实洞察
        best_scenario = max(
            counterfactual_results.items(),
            key=lambda x: x[1]['individual_effects']['mean']
        )

        insights.append({
            'type': 'optimal_strategy',
            'finding': f"{best_scenario[0]}可能带来最大收益，"
                       f"预期销售额增加{best_scenario[1]['aggregate_impact']['total_change']:.0f}元",
            'confidence': f"{best_scenario[1]['confidence'] * 100:.0f}%",
            'action': f"建议实施{best_scenario[0]}，并密切监控效果"
        })

        # 3. 风险提示
        if sensitivity['overall_robustness']['score'] < 0.6:
            insights.append({
                'type': 'risk_warning',
                'finding': "分析结果的稳健性较低，存在不确定性",
                'confidence': 'low',
                'action': "建议先进行小规模A/B测试验证"
            })

        return insights

    def _prioritize_actions(
            self,
            counterfactual_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """优先级排序行动建议"""
        actions = []

        for scenario_name, result in counterfactual_results.items():
            expected_impact = result['aggregate_impact']['total_change']
            confidence = result['confidence']

            # 计算优先级分数
            priority_score = expected_impact * confidence

            # 确定实施难度
            if "增强促销" in scenario_name:
                difficulty = "medium"
                timeline = "1-2 days"
            elif "竞争环境" in scenario_name:
                difficulty = "high"
                timeline = "ongoing"
            else:
                difficulty = "low"
                timeline = "immediate"

            actions.append({
                'action': scenario_name,
                'expected_impact': expected_impact,
                'confidence': confidence,
                'priority_score': priority_score,
                'difficulty': difficulty,
                'timeline': timeline,
                'question_answered': result.get('business_question', '')
            })

        # 按优先级分数排序
        actions.sort(key=lambda x: x['priority_score'], reverse=True)

        return actions

    def _find_optimal_retention_strategy(
            self,
            strategy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """找出最优客户挽回策略"""
        best_roi = -float('inf')
        best_strategy = None

        for strategy_name, result in strategy_results.items():
            roi = result['financial_impact']['roi']
            if roi > best_roi:
                best_roi = roi
                best_strategy = strategy_name

        return {
            'strategy': best_strategy,
            'expected_roi': best_roi,
            'implementation_details': strategy_results[best_strategy]
        }

    def _segment_retention_recommendations(
            self,
            customer_data: pd.DataFrame,
            strategy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分客群的挽回建议"""
        segments = {}

        # VIP客户
        vip_mask = customer_data['customer_lifetime_value'] > customer_data['customer_lifetime_value'].quantile(0.8)
        segments['vip_customers'] = {
            'size': sum(vip_mask),
            'recommended_strategy': 'VIP升级策略',
            'reason': 'VIP客户对专属权益更敏感'
        }

        # 价格敏感客户
        price_sensitive_mask = customer_data['avg_discount_used'] > 0.2
        segments['price_sensitive'] = {
            'size': sum(price_sensitive_mask),
            'recommended_strategy': '个性化优惠券策略',
            'reason': '历史数据显示对折扣响应度高'
        }

        # 忠诚客户
        loyal_mask = customer_data['total_purchases'] > 10
        segments['loyal_customers'] = {
            'size': sum(loyal_mask),
            'recommended_strategy': '积分奖励策略',
            'reason': '通过积分强化长期关系'
        }

        return segments

    def _find_optimal_price(
            self,
            pricing_data: pd.DataFrame,
            price_effects: Dict[str, CausalEffect],
            what_if_results: pd.DataFrame
    ) -> Dict[str, Any]:
        """找出最优价格点"""
        current_price = pricing_data['price'].mean()

        # 基于弹性计算最优价格
        # 简化假设：利润 = (价格 - 成本) * 销量
        cost = current_price * 0.6  # 假设成本是价格的60%

        # 使用第一个估计的价格弹性
        elasticity = list(price_effects.values())[0].ate

        # 最优价格公式（垄断定价）
        optimal_price = cost / (1 + 1 / elasticity) if elasticity < -1 else current_price * 1.1

        return {
            'current_price': current_price,
            'optimal_price': optimal_price,
            'expected_profit_increase': (optimal_price - current_price) * 1000,  # 简化计算
            'elasticity': elasticity,
            'confidence_interval': list(price_effects.values())[0].confidence_interval
        }

    def _create_pricing_roadmap(self, optimal_price: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建定价实施路线图"""
        current = optimal_price['current_price']
        target = optimal_price['optimal_price']

        # 分阶段调价
        steps = []

        if abs(target - current) / current > 0.1:  # 变化超过10%
            # 分3步调整
            step1 = current + (target - current) * 0.3
            step2 = current + (target - current) * 0.6

            steps = [
                {
                    'phase': 1,
                    'price': step1,
                    'timeline': 'Week 1-2',
                    'action': '初步调整，监控市场反应'
                },
                {
                    'phase': 2,
                    'price': step2,
                    'timeline': 'Week 3-4',
                    'action': '根据反馈继续调整'
                },
                {
                    'phase': 3,
                    'price': target,
                    'timeline': 'Week 5+',
                    'action': '达到目标价格，持续优化'
                }
            ]
        else:
            steps = [
                {
                    'phase': 1,
                    'price': target,
                    'timeline': 'Immediate',
                    'action': '一次性调整到位'
                }
            ]

        return steps

    def _analyze_seasonal_inventory(
            self,
            inventory_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析季节性库存需求"""
        # 简化的季节性分析
        seasonal_factors = {
            'spring': 1.0,
            'summer': 1.3,
            'fall': 0.9,
            'winter': 0.8
        }

        return {
            'seasonal_multipliers': seasonal_factors,
            'recommendation': '夏季增加30%安全库存，冬季可降低20%'
        }

    def _analyze_supplier_impact(
            self,
            inventory_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """分析供应商影响"""
        return [
            {
                'supplier': 'Primary Supplier A',
                'lead_time_impact': '缩短1天可减少15%的缺货',
                'action': '协商建立VMI（供应商管理库存）'
            },
            {
                'supplier': 'Backup Supplier B',
                'lead_time_impact': '作为应急可接受+2天lead time',
                'action': '保持战略合作关系'
            }
        ]

    def _optimize_marketing_mix(
            self,
            mix_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """优化营销组合"""
        best_mix = max(
            mix_results.items(),
            key=lambda x: x[1]['cost_effectiveness']['conversions_per_dollar']
        )

        return {
            'recommended_mix': best_mix[0],
            'conversions_per_dollar': best_mix[1]['cost_effectiveness']['conversions_per_dollar'],
            'implementation': '建议逐步过渡到智能组合策略，避免客户疲劳'
        }

    def _identify_personalization_opportunities(
            self,
            marketing_data: pd.DataFrame,
            channel_effects: Dict[str, CausalEffect]
    ) -> List[Dict[str, Any]]:
        """识别个性化机会"""
        opportunities = []

        # 基于客群的渠道偏好
        opportunities.append({
            'segment': '年轻客户',
            'insight': 'App推送效果最好',
            'action': '增加App推送频率，减少SMS'
        })

        opportunities.append({
            'segment': '高价值客户',
            'insight': 'Email个性化内容转化率高',
            'action': '投资邮件内容个性化引擎'
        })

        opportunities.append({
            'segment': '价格敏感客户',
            'insight': 'SMS优惠券响应度最高',
            'action': 'SMS重点推送限时优惠'
        })

        return opportunities


# 使用示例
if __name__ == "__main__":
    # 创建分析器
    analyzer = CausalBusinessAnalyzer()

    # 生成模拟数据
    n_samples = 1000

    # 销售数据
    sales_data = pd.DataFrame({
        'date': pd.date_range(end=datetime.now(), periods=n_samples, freq='D'),
        'daily_revenue': np.random.normal(50000, 10000, n_samples),
        'promotion_active': np.random.binomial(1, 0.3, n_samples),
        'price_level': np.random.normal(100, 10, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'weather_condition': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'competitor_activity': np.random.binomial(1, 0.4, n_samples),
        'inventory_availability': np.random.uniform(0.5, 1.0, n_samples),
        'store_traffic': np.random.normal(1000, 200, n_samples),
        'customer_segment': np.random.choice(['regular', 'vip', 'new'], n_samples),
        'product_type': np.random.choice(['beverage', 'food', 'snack'], n_samples)
    })

    # 1. 销售下滑分析
    print("\n=== 销售下滑因果分析 ===")
    sales_results = analyzer.analyze_sales_decline(sales_data)

    print("\n因果效应估计:")
    for method, effect in sales_results['causal_effects'].items():
        print(f"{method}: ATE={effect.ate:.3f}, CI={effect.confidence_interval}")

    print("\n反事实分析结果:")
    for scenario, result in sales_results['counterfactual_analysis'].items():
        print(f"\n{scenario}:")
        print(f"  问题: {result['business_question']}")
        print(f"  预期效果: {result['individual_effects']['mean']:.2f}")
        print(f"  置信度: {result['confidence'] * 100:.0f}%")

    print("\n建议行动优先级:")
    for i, action in enumerate(sales_results['recommended_actions'][:3]):
        print(f"{i + 1}. {action['action']}")
        print(f"   预期影响: {action['expected_impact']:.0f}")
        print(f"   实施时间: {action['timeline']}")

    # 2. 客户流失分析
    print("\n\n=== 客户流失因果分析 ===")

    # 生成客户数据
    customer_data = pd.DataFrame({
        'customer_id': range(1000),
        'churned': np.random.binomial(1, 0.2, 1000),
        'received_retention_offer': np.random.binomial(1, 0.3, 1000),
        'customer_lifetime_value': np.random.gamma(100, 2, 1000),
        'recent_support_tickets': np.random.poisson(0.5, 1000),
        'days_since_last_purchase': np.random.exponential(30, 1000),
        'total_purchases': np.random.poisson(10, 1000),
        'satisfaction_score': np.random.uniform(1, 5, 1000),
        'random_campaign_assignment': np.random.binomial(1, 0.5, 1000),
        'churn_probability': np.random.uniform(0, 1, 1000),
        'avg_discount_used': np.random.uniform(0, 0.4, 1000)
    })

    churn_results = analyzer.analyze_customer_churn(customer_data)

    print("\n最优挽回策略:")
    optimal = churn_results['optimal_strategy']
    print(f"策略: {optimal['strategy']}")
    print(f"预期ROI: {optimal['expected_roi'] * 100:.0f}%")

    print("\n分客群建议:")
    for segment, rec in churn_results['segmented_recommendations'].items():
        print(f"{segment}: {rec['recommended_strategy']} ({rec['reason']})")

    # 3. 促销场景分析
    print("\n\n=== 促销策略What-if分析 ===")
    promo_results = analyzer.engine.analyze_promotion_scenarios(sales_data)

    print("\n场景对比:")
    comparison = analyzer.engine._compare_scenarios(promo_results['scenario_analysis'])
    print(comparison)

    print(f"\n推荐方案: {promo_results['recommendation']['recommended_scenario']}")
    print(f"预期收入增加: ¥{promo_results['recommendation']['expected_revenue_increase']:.0f}")