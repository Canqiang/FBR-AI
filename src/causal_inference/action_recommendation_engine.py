"""
UMe 茶饮因果分析行动推荐引擎
根据因果分析结果自动生成具体的行动建议
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json


class UMeActionRecommendationEngine:
    """UMe行动推荐引擎 - 将因果分析结果转换为具体行动"""

    def __init__(self):
        self.attribution_action_mapping = self._load_attribution_mapping()
        self.action_templates = self._load_action_templates()
        self.priority_weights = self._load_priority_weights()

    def _load_attribution_mapping(self) -> Dict[str, Dict]:
        """加载归因-行动映射关系"""
        return {
            'promotion_negative': {
                'attribution': '促销活动造成负面影响',
                'root_causes': [
                    '过度促销导致品牌价值稀释',
                    '促销成本超过收益',
                    '促销时机不当（如高温天促销冷饮）'
                ],
                'linked_actions': [
                    'promotion_adjustment',
                    'promotion_timing_optimization',
                    'value_based_marketing',
                    'price_strategy_review'
                ]
            },
            'promotion_positive': {
                'attribution': '促销活动带来正面效果',
                'root_causes': [
                    '促销策略有效提升销量',
                    '客户对促销响应积极',
                    '促销ROI良好'
                ],
                'linked_actions': [
                    'promotion_expansion',
                    'promotion_optimization',
                    'customer_segmentation_promo'
                ]
            },
            'weather_negative': {
                'attribution': '天气因素造成销量下降',
                'root_causes': [
                    '恶劣天气减少客流',
                    '温度不适合当前产品',
                    '降水影响外出消费'
                ],
                'linked_actions': [
                    'weather_adaptive_marketing',
                    'delivery_enhancement',
                    'indoor_experience_improvement',
                    'weather_specific_products'
                ]
            },
            'weather_positive': {
                'attribution': '天气因素促进销量增长',
                'root_causes': [
                    '适宜天气增加客流',
                    '温度适合产品消费',
                    '季节性需求增长'
                ],
                'linked_actions': [
                    'weather_opportunity_marketing',
                    'inventory_preparation',
                    'staff_optimization'
                ]
            },
            'weekend_positive': {
                'attribution': '周末效应显著提升营收',
                'root_causes': [
                    '周末客流量大幅增加',
                    '消费者休闲时间充足',
                    '社交消费需求增长'
                ],
                'linked_actions': [
                    'weekend_capacity_expansion',
                    'weekend_marketing_intensify',
                    'weekend_experience_enhancement'
                ]
            },
            'holiday_positive': {
                'attribution': '节假日带来营收提升',
                'root_causes': [
                    '节假日消费意愿增强',
                    '家庭聚会需求增加',
                    '庆祝活动带动消费'
                ],
                'linked_actions': [
                    'holiday_special_campaign',
                    'family_package_promotion',
                    'festive_marketing'
                ]
            },
            'interaction_negative': {
                'attribution': '因素组合产生负面交互效应',
                'root_causes': [
                    '多个策略相互冲突',
                    '资源配置不合理',
                    '时机选择错误'
                ],
                'linked_actions': [
                    'strategy_coordination',
                    'timing_optimization',
                    'resource_reallocation'
                ]
            },
            'heterogeneity_identified': {
                'attribution': '不同店铺/条件下效果差异显著',
                'root_causes': [
                    '店铺特性差异',
                    '地区消费习惯不同',
                    '竞争环境差异'
                ],
                'linked_actions': [
                    'store_specific_strategy',
                    'localized_marketing',
                    'performance_benchmarking'
                ]
            }
        }

    def _load_action_templates(self) -> Dict[str, Dict]:
        """加载行动模板"""
        return {
            'promotion_adjustment': {
                'name': '促销策略调整',
                'template': '建议调整{promotion_type}促销策略：{specific_adjustment}',
                'urgency': 'high',
                'timeline': '立即执行',
                'expected_impact': '预计减少${loss_amount}损失',
                'kpi': ['营收增长', '促销ROI', '客户满意度']
            },
            'promotion_timing_optimization': {
                'name': '促销时机优化',
                'template': '优化促销时机：避免在{avoid_conditions}进行促销，建议在{optimal_conditions}时执行',
                'urgency': 'medium',
                'timeline': '1-2周内调整',
                'expected_impact': '预计提升促销效果{improvement_percentage}%',
                'kpi': ['促销转化率', '客单价', '利润率']
            },
            'value_based_marketing': {
                'name': '价值导向营销',
                'template': '转向价值营销：强调{value_proposition}，减少纯价格竞争',
                'urgency': 'medium',
                'timeline': '2-4周执行',
                'expected_impact': '预计提升品牌价值感知',
                'kpi': ['品牌认知', '客户忠诚度', '复购率']
            },
            'weather_adaptive_marketing': {
                'name': '天气自适应营销',
                'template': '{weather_condition}天气营销策略：{weather_strategy}',
                'urgency': 'high',
                'timeline': '根据天气预报执行',
                'expected_impact': '预计减少天气负面影响{impact_reduction}%',
                'kpi': ['日销量稳定性', '客流量', '季节适应性']
            },
            'delivery_enhancement': {
                'name': '配送服务增强',
                'template': '恶劣天气配送优化：{delivery_improvements}',
                'urgency': 'medium',
                'timeline': '1个月内完善',
                'expected_impact': '预计恶劣天气销量提升{sales_lift}%',
                'kpi': ['外卖订单量', '客户满意度', '配送时效']
            },
            'weekend_capacity_expansion': {
                'name': '周末产能扩张',
                'template': '周末增加{capacity_type}：{specific_expansion}',
                'urgency': 'high',
                'timeline': '2周内实施',
                'expected_impact': '预计周末营收提升${weekend_revenue_increase}',
                'kpi': ['周末销量', '客户等待时间', '服务质量']
            },
            'holiday_special_campaign': {
                'name': '节假日特别活动',
                'template': '{holiday_name}特别营销：{campaign_details}',
                'urgency': 'high',
                'timeline': '节前2周准备',
                'expected_impact': '预计节假日营收提升{holiday_lift}%',
                'kpi': ['节假日销量', '新客获取', '品牌曝光']
            },
            'store_specific_strategy': {
                'name': '店铺定制化策略',
                'template': '为{store_segment}制定专属策略：{customized_strategy}',
                'urgency': 'medium',
                'timeline': '1个月内部署',
                'expected_impact': '预计低效店铺性能提升{performance_improvement}%',
                'kpi': ['店铺间绩效差异', '整体营收', '运营效率']
            },
            'strategy_coordination': {
                'name': '策略协调优化',
                'template': '优化策略组合：{coordination_plan}',
                'urgency': 'high',
                'timeline': '立即调整',
                'expected_impact': '预计消除${negative_interaction}负面交互损失',
                'kpi': ['策略协同效应', '资源利用率', '整体ROI']
            }
        }

    def _load_priority_weights(self) -> Dict[str, float]:
        """加载优先级权重"""
        return {
            'financial_impact': 0.4,  # 财务影响权重
            'urgency': 0.3,  # 紧急程度权重
            'feasibility': 0.2,  # 可执行性权重
            'strategic_importance': 0.1  # 战略重要性权重
        }

    def analyze_and_recommend(self, causal_results: Dict[str, Any],
                              data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """基于因果分析结果生成行动推荐"""

        print("🎯 正在生成行动推荐...")

        # 1. 分析因果结果，识别问题和机会
        attributions = self._identify_attributions(causal_results)

        # 2. 为每个归因生成具体行动
        action_recommendations = []

        for attribution in attributions:
            actions = self._generate_actions_for_attribution(attribution, data_summary)
            action_recommendations.extend(actions)

        # 3. 对行动进行优先级排序
        prioritized_actions = self._prioritize_actions(action_recommendations, data_summary)

        # 4. 生成执行计划
        execution_plan = self._create_execution_plan(prioritized_actions)

        return {
            'attributions': attributions,
            'recommended_actions': prioritized_actions,
            'execution_plan': execution_plan,
            'summary': self._create_recommendation_summary(prioritized_actions)
        }

    def _identify_attributions(self, causal_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别归因"""
        attributions = []

        # 分析主要因果效应
        for factor_name, result in causal_results.items():
            if factor_name in ['interactions', 'heterogeneity']:
                continue

            if 'ate' not in result or 'error' in result:
                continue

            ate = result['ate']
            significant = result.get('significant', False)

            # 促销效应分析
            if factor_name == 'has_promotion':
                if significant and ate < -50:  # 显著负面效应
                    attributions.append({
                        'type': 'promotion_negative',
                        'factor': factor_name,
                        'impact': ate,
                        'significance': significant,
                        'severity': 'high' if ate < -100 else 'medium'
                    })
                elif significant and ate > 50:  # 显著正面效应
                    attributions.append({
                        'type': 'promotion_positive',
                        'factor': factor_name,
                        'impact': ate,
                        'significance': significant,
                        'opportunity': 'high' if ate > 150 else 'medium'
                    })

            # 天气效应分析
            elif factor_name in ['is_hot', 'is_rainy', 'is_cold']:
                if significant and ate < -30:
                    attributions.append({
                        'type': 'weather_negative',
                        'factor': factor_name,
                        'impact': ate,
                        'significance': significant,
                        'weather_type': factor_name.replace('is_', '')
                    })
                elif significant and ate > 30:
                    attributions.append({
                        'type': 'weather_positive',
                        'factor': factor_name,
                        'impact': ate,
                        'significance': significant,
                        'weather_type': factor_name.replace('is_', '')
                    })

            # 周末效应
            elif factor_name == 'is_weekend' and significant and ate > 50:
                attributions.append({
                    'type': 'weekend_positive',
                    'factor': factor_name,
                    'impact': ate,
                    'significance': significant
                })

            # 节假日效应
            elif factor_name == 'is_holiday' and significant and ate > 50:
                attributions.append({
                    'type': 'holiday_positive',
                    'factor': factor_name,
                    'impact': ate,
                    'significance': significant
                })

        # 分析交互效应
        interactions = causal_results.get('interactions', {})
        for interaction_key, interaction_data in interactions.items():
            if 'error' in interaction_data:
                continue

            interaction_effect = interaction_data.get('interaction_effect', 0)

            if abs(interaction_effect) > 50:  # 显著交互效应
                attributions.append({
                    'type': 'interaction_negative' if interaction_effect < 0 else 'interaction_positive',
                    'factors': interaction_key.split('_x_'),
                    'impact': interaction_effect,
                    'interaction_key': interaction_key
                })

        # 分析异质性
        heterogeneity = causal_results.get('heterogeneity', {})
        if heterogeneity:
            # 检查店铺间差异
            store_effects = heterogeneity.get('promotion_by_store', {})
            if store_effects and len(store_effects) > 1:
                effects = [data['effect'] for data in store_effects.values()]
                if np.std(effects) > 100:  # 标准差大于100，说明差异显著
                    attributions.append({
                        'type': 'heterogeneity_identified',
                        'dimension': 'store',
                        'variance': np.std(effects),
                        'details': store_effects
                    })

        return attributions

    def _generate_actions_for_attribution(self, attribution: Dict[str, Any],
                                          data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为单个归因生成行动建议"""

        attribution_type = attribution['type']
        mapping = self.attribution_action_mapping.get(attribution_type, {})
        linked_actions = mapping.get('linked_actions', [])

        actions = []

        for action_key in linked_actions:
            action_template = self.action_templates.get(action_key, {})
            if not action_template:
                continue

            # 生成具体的行动建议
            action = self._customize_action(action_template, attribution, data_summary)
            actions.append(action)

        return actions

    def _customize_action(self, template: Dict[str, Any], attribution: Dict[str, Any],
                          data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """定制化行动建议"""

        action = template.copy()

        # 基于归因类型定制内容
        if attribution['type'] == 'promotion_negative':
            impact = attribution['impact']
            if action['name'] == '促销策略调整':
                action[
                    'description'] = f"当前促销策略造成${abs(impact):.0f}的负面影响。建议：1）暂停在高需求时段的促销；2）调整促销幅度从价格导向转为价值导向；3）重新评估促销ROI。"
                action['financial_impact'] = abs(impact) * 30  # 按月计算潜在影响

        elif attribution['type'] == 'weather_negative':
            weather_type = attribution.get('weather_type', 'bad_weather')
            impact = attribution['impact']
            if action['name'] == '天气自适应营销':
                weather_strategies = {
                    'rainy': '推出"雨天优惠外卖"、"室内温馨体验"套餐',
                    'hot': '强化冷饮推广、延长营业时间、增加空调舒适度',
                    'cold': '推出热饮套餐、室内保暖体验、热食组合'
                }
                strategy = weather_strategies.get(weather_type, '制定天气应对策略')
                action['description'] = f"{weather_type.title()}天气造成${abs(impact):.0f}营收损失。建议策略：{strategy}"
                action['financial_impact'] = abs(impact) * 30

        elif attribution['type'] == 'weekend_positive':
            impact = attribution['impact']
            if action['name'] == '周末产能扩张':
                action[
                    'description'] = f"周末效应显著，每天可带来${impact:.0f}额外营收。建议：1）增加周末员工排班；2）延长周末营业时间；3）推出周末专属产品；4）优化周末供应链。"
                action['financial_impact'] = impact * 8  # 按月4个周末计算

        elif attribution['type'] == 'interaction_negative':
            impact = attribution['impact']
            factors = attribution.get('factors', [])
            if action['name'] == '策略协调优化':
                action[
                    'description'] = f"{' × '.join(factors)}组合产生${abs(impact):.0f}负面交互效应。建议：1）避免同时执行这些策略；2）重新安排策略时机；3）优化资源分配。"
                action['financial_impact'] = abs(impact) * 30

        # 设置执行参数
        action['attribution_id'] = f"{attribution['type']}_{hash(str(attribution)) % 10000}"
        action['created_at'] = datetime.now().isoformat()
        action['status'] = 'pending'

        return action

    def _prioritize_actions(self, actions: List[Dict[str, Any]],
                            data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对行动建议进行优先级排序"""

        for action in actions:
            # 计算综合得分
            financial_score = min(action.get('financial_impact', 0) / 1000, 10)  # 财务影响得分
            urgency_score = {'high': 10, 'medium': 6, 'low': 3}.get(action.get('urgency', 'medium'), 6)
            feasibility_score = 8  # 默认可执行性得分
            strategic_score = 7  # 默认战略重要性得分

            # 加权计算总分
            weights = self.priority_weights
            total_score = (
                    financial_score * weights['financial_impact'] +
                    urgency_score * weights['urgency'] +
                    feasibility_score * weights['feasibility'] +
                    strategic_score * weights['strategic_importance']
            )

            action['priority_score'] = total_score
            action['priority_level'] = 'high' if total_score > 8 else 'medium' if total_score > 5 else 'low'

        # 按得分排序
        return sorted(actions, key=lambda x: x['priority_score'], reverse=True)

    def _create_execution_plan(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建执行计划"""

        # 按优先级分组
        high_priority = [a for a in actions if a['priority_level'] == 'high']
        medium_priority = [a for a in actions if a['priority_level'] == 'medium']
        low_priority = [a for a in actions if a['priority_level'] == 'low']

        # 按时间线分组
        immediate = [a for a in actions if '立即' in a.get('timeline', '')]
        short_term = [a for a in actions if '周' in a.get('timeline', '')]
        medium_term = [a for a in actions if '月' in a.get('timeline', '')]

        return {
            'priority_phases': {
                'phase_1_critical': high_priority[:3],  # 最多3个高优先级行动
                'phase_2_important': medium_priority[:4],  # 最多4个中优先级行动
                'phase_3_improvement': low_priority[:3]  # 最多3个低优先级行动
            },
            'timeline_phases': {
                'immediate_actions': immediate,
                'short_term_actions': short_term,
                'medium_term_actions': medium_term
            },
            'estimated_total_impact': sum([a.get('financial_impact', 0) for a in actions[:5]]),  # 前5个行动的预期影响
            'recommended_focus': high_priority[0]['name'] if high_priority else '持续监控和数据收集'
        }

    def _create_recommendation_summary(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建推荐摘要"""

        total_actions = len(actions)
        high_priority_count = len([a for a in actions if a['priority_level'] == 'high'])
        total_potential_impact = sum([a.get('financial_impact', 0) for a in actions[:5]])

        # 生成执行摘要
        if high_priority_count > 0:
            top_action = actions[0]
            key_recommendation = f"立即执行'{top_action['name']}'，预期月度影响${top_action.get('financial_impact', 0):.0f}"
        else:
            key_recommendation = "继续监控关键指标，寻找优化机会"

        return {
            'total_actions': total_actions,
            'high_priority_actions': high_priority_count,
            'estimated_monthly_impact': total_potential_impact,
            'key_recommendation': key_recommendation,
            'next_review_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'success_metrics': [
                '营收增长率',
                '促销ROI',
                '客户满意度',
                '运营效率',
                '市场份额'
            ]
        }

    def format_action_report(self, recommendations: Dict[str, Any]) -> str:
        """格式化行动报告"""

        report_lines = []
        report_lines.append("# 🎯 UMe 茶饮行动推荐报告")
        report_lines.append("")

        # 执行摘要
        summary = recommendations['summary']
        report_lines.append("## 📊 执行摘要")
        report_lines.append(f"- **推荐行动数**: {summary['total_actions']}个")
        report_lines.append(f"- **高优先级行动**: {summary['high_priority_actions']}个")
        report_lines.append(f"- **预期月度影响**: ${summary['estimated_monthly_impact']:,.0f}")
        report_lines.append(f"- **核心建议**: {summary['key_recommendation']}")
        report_lines.append("")

        # 立即行动建议
        execution_plan = recommendations['execution_plan']
        critical_actions = execution_plan['priority_phases']['phase_1_critical']

        if critical_actions:
            report_lines.append("## 🔥 立即行动 (关键优先级)")
            for i, action in enumerate(critical_actions, 1):
                report_lines.append(f"### {i}. {action['name']}")
                report_lines.append(f"**描述**: {action.get('description', '待详细制定')}")
                report_lines.append(f"**预期影响**: ${action.get('financial_impact', 0):,.0f}/月")
                report_lines.append(f"**执行时间**: {action.get('timeline', '待确定')}")
                report_lines.append(f"**关键指标**: {', '.join(action.get('kpi', []))}")
                report_lines.append("")

        # 短期行动建议
        important_actions = execution_plan['priority_phases']['phase_2_important']
        if important_actions:
            report_lines.append("## ⭐ 短期优化 (重要优先级)")
            for i, action in enumerate(important_actions, 1):
                report_lines.append(f"### {i}. {action['name']}")
                report_lines.append(f"**描述**: {action.get('description', '待详细制定')}")
                report_lines.append(f"**预期影响**: ${action.get('financial_impact', 0):,.0f}/月")
                report_lines.append("")

        # 长期改进建议
        improvement_actions = execution_plan['priority_phases']['phase_3_improvement']
        if improvement_actions:
            report_lines.append("## 💡 长期改进 (优化优先级)")
            for i, action in enumerate(improvement_actions, 1):
                report_lines.append(f"- **{action['name']}**: {action.get('description', '待详细制定')}")
            report_lines.append("")

        # 监控建议
        report_lines.append("## 📈 成功监控指标")
        for metric in summary['success_metrics']:
            report_lines.append(f"- {metric}")
        report_lines.append("")

        report_lines.append(f"## 📅 下次回顾时间")
        report_lines.append(f"{summary['next_review_date']} - 建议每月回顾行动执行效果")

        return "\n".join(report_lines)


# ============================================================================
# 使用示例和测试
# ============================================================================
#
# def test_action_recommendation():
#     """测试行动推荐功能"""
#
#     # 模拟因果分析结果
#     mock_causal_results = {
#         'has_promotion': {
#             'ate': -850,  # 促销造成负面影响
#             'ci_lower': -1200,
#             'ci_upper': -500,
#             'significant': True,
#             'treatment_rate': 0.35
#         },
#         'is_weekend': {
#             'ate': 245,  # 周末正面效应
#             'ci_lower': 180,
#             'ci_upper': 310,
#             'significant': True,
#             'treatment_rate': 0.29
#         },
#         'is_hot': {
#             'ate': 120,  # 高温正面效应
#             'ci_lower': 50,
#             'ci_upper': 190,
#             'significant': True,
#             'treatment_rate': 0.25
#         },
#         'interactions': {
#             'is_hot_x_has_promotion': {
#                 'interaction_effect': -1021,  # 您提到的负面交互效应
#                 'factor1_main_effect': 120,
#                 'factor2_main_effect': -850,
#                 'combined_effect': -1751
#             }
#         },
#         'heterogeneity': {
#             'promotion_by_store': {
#                 'STORE_A': {'effect': -200, 'sample_size': 45},
#                 'STORE_B': {'effect': 50, 'sample_size': 38},
#                 'STORE_C': {'effect': -500, 'sample_size': 42}
#             }
#         }
#     }
#
#     mock_data_summary = {
#         'total_records': 1500,
#         'stores_count': 8,
#         'average_daily_revenue': 2500
#     }
#
#     # 创建推荐引擎
#     engine = UMeActionRecommendationEngine()
#
#     # 生成推荐
#     recommendations = engine.analyze_and_recommend(mock_causal_results, mock_data_summary)
#
#     # 生成报告
#     report = engine.format_action_report(recommendations)
#
#     return recommendations, report
#
#
# if __name__ == "__main__":
#     print("🚀 测试 UMe 行动推荐引擎")
#     print("=" * 60)
#
#     recommendations, report = test_action_recommendation()
#
#     print("📊 推荐结果:")
#     print(f"- 识别归因: {len(recommendations['attributions'])}个")
#     print(f"- 推荐行动: {len(recommendations['recommended_actions'])}个")
#     print(f"- 高优先级行动: {recommendations['summary']['high_priority_actions']}个")
#     print(f"- 预期月度影响: ${recommendations['summary']['estimated_monthly_impact']:,.0f}")
#
#     print("\n" + "=" * 60)
#     print("📋 详细行动报告:")
#     print("=" * 60)
#     print(report)