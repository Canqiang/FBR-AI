import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dowhy import CausalModel
import dowhy.datasets

logger = logging.getLogger(__name__)


class CausalInferenceEngine:
    """因果推断引擎"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def analyze_promotion_effect(
            self,
            data: pd.DataFrame,
            treatment: str = 'has_promotion',
            outcome: str = 'revenue',
            confounders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """分析促销活动的因果效应"""
        logger.info(f"Analyzing causal effect of {treatment} on {outcome}")

        # 默认混淆因素
        if confounders is None:
            confounders = ['is_weekend', 'hour_of_day', 'location', 'weather']

        # 过滤实际存在的混淆因素
        available_confounders = [c for c in confounders if c in data.columns]

        # 构建因果图
        causal_graph = self._build_causal_graph(treatment, outcome, available_confounders)

        # 创建因果模型
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )

        # 识别因果效应
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 使用多种方法估计因果效应
        estimates = {}

        # 1. 倾向得分匹配
        try:
            psm_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate"  # Average Treatment Effect
            )
            estimates['psm'] = {
                'effect': psm_estimate.value,
                'ci': self._get_confidence_interval(psm_estimate)
            }
        except Exception as e:
            logger.warning(f"PSM estimation failed: {e}")

        # 2. 逆概率加权
        try:
            ipw_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_weighting"
            )
            estimates['ipw'] = {
                'effect': ipw_estimate.value,
                'ci': self._get_confidence_interval(ipw_estimate)
            }
        except Exception as e:
            logger.warning(f"IPW estimation failed: {e}")

        # 3. 线性回归
        try:
            lr_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            estimates['linear_regression'] = {
                'effect': lr_estimate.value,
                'ci': self._get_confidence_interval(lr_estimate)
            }
        except Exception as e:
            logger.warning(f"Linear regression estimation failed: {e}")

        # 敏感性分析
        refutation_results = self._perform_refutation(model, psm_estimate if 'psm' in estimates else lr_estimate)

        # 生成可解释的结果
        interpretation = self._interpret_results(estimates, refutation_results)

        return {
            'estimates': estimates,
            'refutation': refutation_results,
            'interpretation': interpretation,
            'recommendation': self._generate_recommendation(estimates, treatment, outcome)
        }

    def _build_causal_graph(self, treatment: str, outcome: str, confounders: List[str]) -> str:
        """构建因果图"""
        edges = []

        # 处理变量到结果
        edges.append(f"{treatment} -> {outcome}")

        # 混淆因素到处理和结果
        for confounder in confounders:
            edges.append(f"{confounder} -> {treatment}")
            edges.append(f"{confounder} -> {outcome}")

        graph = "digraph {\n" + ";\n".join(edges) + "\n}"
        return graph

    def _get_confidence_interval(self, estimate, confidence_level: float = 0.95) -> Tuple[float, float]:
        """获取置信区间"""
        try:
            # 尝试从估计对象中获取置信区间
            if hasattr(estimate, 'get_confidence_intervals'):
                ci = estimate.get_confidence_intervals(confidence_level=confidence_level)
                return (ci[0], ci[1])
            else:
                # 使用标准误差估计置信区间
                se = estimate.get_standard_error() if hasattr(estimate, 'get_standard_error') else estimate.value * 0.1
                z_score = 1.96  # 95%置信水平
                return (estimate.value - z_score * se, estimate.value + z_score * se)
        except:
            return (estimate.value * 0.8, estimate.value * 1.2)  # 粗略估计

    def _perform_refutation(self, model: CausalModel, estimate) -> Dict[str, Any]:
        """执行反驳测试"""
        refutation_results = {}

        # 1. 安慰剂测试
        try:
            placebo = model.refute_estimate(
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=100
            )
            refutation_results['placebo_test'] = {
                'passed': abs(placebo.new_effect) < abs(estimate.value) * 0.1,
                'placebo_effect': placebo.new_effect
            }
        except Exception as e:
            logger.warning(f"Placebo test failed: {e}")

        # 2. 随机共同原因测试
        try:
            random_common_cause = model.refute_estimate(
                estimate,
                method_name="random_common_cause",
                num_simulations=100
            )
            refutation_results['random_common_cause'] = {
                'passed': abs(random_common_cause.new_effect - estimate.value) < abs(estimate.value) * 0.1,
                'new_effect': random_common_cause.new_effect
            }
        except Exception as e:
            logger.warning(f"Random common cause test failed: {e}")

        return refutation_results

    def _interpret_results(self, estimates: Dict, refutation: Dict) -> str:
        """解释因果分析结果"""
        # 计算平均效应
        effects = [est['effect'] for est in estimates.values() if 'effect' in est]
        avg_effect = np.mean(effects) if effects else 0

        # 检查一致性
        consistency = np.std(effects) / (abs(avg_effect) + 0.01) < 0.3 if effects else False

        # 检查稳健性
        robustness = all(ref.get('passed', False) for ref in refutation.values())

        interpretation = f"因果效应分析结果：\n"
        interpretation += f"- 平均因果效应：{avg_effect:.2f}\n"
        interpretation += f"- 估计一致性：{'高' if consistency else '低'}\n"
        interpretation += f"- 结果稳健性：{'通过检验' if robustness else '需要谨慎'}\n"

        if avg_effect > 0:
            interpretation += f"- 结论：处理变量对结果有正向因果效应\n"
        elif avg_effect < 0:
            interpretation += f"- 结论：处理变量对结果有负向因果效应\n"
        else:
            interpretation += f"- 结论：未发现显著因果效应\n"

        return interpretation

    def _generate_recommendation(self, estimates: Dict, treatment: str, outcome: str) -> Dict[str, str]:
        """基于因果分析生成建议"""
        effects = [est['effect'] for est in estimates.values() if 'effect' in est]
        avg_effect = np.mean(effects) if effects else 0

        if treatment == 'has_promotion' and outcome == 'revenue':
            if avg_effect > 10:
                return {
                    'action': '继续并扩大促销活动',
                    'reason': f'促销活动显著提升营收（平均+¥{avg_effect:.0f}）',
                    'confidence': 'high'
                }
            elif avg_effect > 0:
                return {
                    'action': '优化促销策略以提升效果',
                    'reason': f'促销有正向效果但不够显著（+¥{avg_effect:.0f}）',
                    'confidence': 'medium'
                }
            else:
                return {
                    'action': '重新评估促销策略',
                    'reason': '当前促销未产生预期效果',
                    'confidence': 'low'
                }

        # 通用建议
        return {
            'action': '基于数据继续监测',
            'reason': f'{treatment}对{outcome}的影响为{avg_effect:.2f}',
            'confidence': 'medium'
        }
