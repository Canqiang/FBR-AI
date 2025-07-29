import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..data.repositories import OrderRepository, CustomerRepository, AnalyticsRepository
from ..analytics.predictive import SalesPredictorEngine
from ..analytics.causal import CausalInferenceEngine
from ..analytics.anomaly import AnomalyDetectionEngine
from ..analytics.optimization import OptimizationEngine
from ..llm.agents import AIGrowthAgent
from ..llm.chains import AnalysisChains

logger = logging.getLogger(__name__)


class AIGrowthEngineCore:
    """AI增长引擎核心类"""

    def __init__(self):
        # 数据层
        self.order_repo = OrderRepository()
        self.customer_repo = CustomerRepository()
        self.analytics_repo = AnalyticsRepository()

        # 分析引擎
        self.predictor = SalesPredictorEngine()
        self.causal_engine = CausalInferenceEngine()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.optimizer = OptimizationEngine()

        # AI层
        self.ai_agent = AIGrowthAgent()
        self.analysis_chains = AnalysisChains()

        # 状态管理
        self.state = {
            'last_run': None,
            'current_analysis': {},
            'recommendations': []
        }

    def run_daily_analysis(self) -> Dict[str, Any]:
        """运行每日分析"""
        logger.info("Starting daily analysis")
        start_time = datetime.now()

        try:
            # 1. 数据收集
            logger.info("Step 1: Collecting data")
            data = self._collect_daily_data()

            # 2. 异常检测
            logger.info("Step 2: Detecting anomalies")
            anomalies = self._detect_anomalies(data)

            # 3. 预测分析
            logger.info("Step 3: Running predictions")
            predictions = self._run_predictions(data)

            # 4. 因果分析
            logger.info("Step 4: Causal analysis")
            causal_results = self._run_causal_analysis(data)

            # 5. 优化建议
            logger.info("Step 5: Generating optimizations")
            optimizations = self._generate_optimizations(data, predictions)

            # 6. AI洞察生成
            logger.info("Step 6: Generating AI insights")
            insights = self._generate_ai_insights(
                data, anomalies, predictions, causal_results, optimizations
            )

            # 7. 汇总结果
            results = {
                'timestamp': start_time,
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'success',
                'data_summary': self._summarize_data(data),
                'anomalies': anomalies,
                'predictions': predictions,
                'causal_analysis': causal_results,
                'optimizations': optimizations,
                'insights': insights,
                'recommendations': self._prioritize_recommendations(optimizations, insights)
            }

            # 更新状态
            self.state['last_run'] = start_time
            self.state['current_analysis'] = results

            logger.info(f"Daily analysis completed in {results['duration']:.2f} seconds")
            return results

        except Exception as e:
            logger.error(f"Daily analysis failed: {e}", exc_info=True)
            return {
                'timestamp': start_time,
                'status': 'failed',
                'error': str(e)
            }

    def _collect_daily_data(self) -> Dict[str, pd.DataFrame]:
        """收集每日数据"""
        end_date = datetime.now()

        # 不同时间范围的数据
        data = {
            'daily_sales': self.order_repo.get_daily_sales(
                end_date - timedelta(days=90), end_date
            ),
            'item_performance': self.order_repo.get_item_performance(days=30),
            'customer_segments': self.customer_repo.get_customer_segments(),
            'time_series': self.analytics_repo.get_time_series_data(days=180),
            'promotion_effectiveness': self.analytics_repo.get_promotion_effectiveness(days=30)
        }

        return data

    def _detect_anomalies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """检测异常"""
        anomalies = {}

        # 销售异常检测
        if 'daily_sales' in data:
            sales_with_anomalies = self.anomaly_detector.detect_sales_anomalies(
                data['daily_sales']
            )
            anomalies['sales_anomalies'] = sales_with_anomalies[
                sales_with_anomalies['is_anomaly']
            ].to_dict('records')

        # 模式变化检测
        if 'time_series' in data:
            pattern_changes = self.anomaly_detector.detect_pattern_changes(
                data['time_series'],
                metric='y'
            )
            anomalies['pattern_changes'] = pattern_changes

        return anomalies

    def _run_predictions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行预测"""
        predictions = {}

        # 销售预测
        if 'time_series' in data:
            sales_forecast = self.predictor.predict_with_prophet(
                data['time_series'],
                periods=7
            )
            predictions['sales_forecast'] = sales_forecast.to_dict('records')

        # 商品需求预测
        if 'item_performance' in data:
            item_demand = self.predictor.predict_item_demand(
                data['item_performance']
            )
            predictions['item_demand'] = item_demand

        return predictions

    def _run_causal_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行因果分析"""
        causal_results = {}

        # 分析促销效果
        if 'promotion_effectiveness' in data and len(data['promotion_effectiveness']) > 0:
            # 准备数据
            promo_data = data['daily_sales'].copy()
            promo_data['has_promotion'] = promo_data['date'].isin(
                data['promotion_effectiveness']['date']
            ).astype(int)

            causal_results['promotion_effect'] = self.causal_engine.analyze_promotion_effect(
                promo_data,
                treatment='has_promotion',
                outcome='total_revenue'
            )

        return causal_results

    def _generate_optimizations(
            self,
            data: Dict[str, pd.DataFrame],
            predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成优化建议"""
        optimizations = {}

        # 价格优化
        if 'item_performance' in data:
            # 模拟成本数据（实际应从数据库获取）
            item_costs = {
                item: price * 0.4  # 假设成本是售价的40%
                for item, price in data['item_performance'].groupby('item_name')['avg_price'].mean().items()
            }

            pricing_optimization = self.optimizer.optimize_pricing(
                data['item_performance'],
                item_costs
            )
            optimizations['pricing'] = pricing_optimization

        # 库存优化
        if 'item_demand' in predictions:
            # 准备需求预测数据
            demand_df = pd.DataFrame({
                'item_name': data['item_performance']['item_name'].unique()[:10],  # Top 10 items
                'predicted_demand': [100] * 10  # 简化的需求预测
            })

            inventory_optimization = self.optimizer.optimize_inventory(
                demand_df,
                storage_capacity=1000
            )
            optimizations['inventory'] = inventory_optimization

        return optimizations

    def _generate_ai_insights(
            self,
            data: Dict[str, pd.DataFrame],
            anomalies: Dict[str, Any],
            predictions: Dict[str, Any],
            causal_results: Dict[str, Any],
            optimizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成AI洞察"""
        insights = {}

        # 生成日报
        if 'daily_sales' in data:
            today_data = data['daily_sales'].iloc[0] if len(data['daily_sales']) > 0 else {}
            yesterday_data = data['daily_sales'].iloc[1] if len(data['daily_sales']) > 1 else {}

            daily_report_chain = self.analysis_chains.create_daily_report_chain()
            daily_report = daily_report_chain.run(
                date=datetime.now().strftime('%Y-%m-%d'),
                metrics=self._format_metrics(today_data),
                comparison=self._format_comparison(today_data, yesterday_data)
            )
            insights['daily_report'] = daily_report

        # 如果有异常，生成问题诊断
        if anomalies.get('sales_anomalies'):
            problem_chain = self.analysis_chains.create_problem_diagnosis_chain()
            diagnosis = problem_chain.run(
                period="最近7天",
                decline_pct=20,  # 示例
                factors=self._format_anomaly_factors(anomalies),
                historical_context="历史同期平均销售额为X元"
            )
            insights['problem_diagnosis'] = diagnosis

        # 客户洞察
        if 'customer_segments' in data:
            customer_chain = self.analysis_chains.create_customer_insight_chain()
            customer_insights = customer_chain.run(
                segment_data=data['customer_segments'].to_string(),
                behavior_patterns="高价值客户偏好晚餐时段消费",
                trends="新客获取率下降15%"
            )
            insights['customer_insights'] = customer_insights

        return insights

    def _summarize_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """汇总数据"""
        summary = {}

        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                summary[key] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                        'end': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
                    }
                }

        return summary

    def _prioritize_recommendations(
            self,
            optimizations: Dict[str, Any],
            insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """优先级排序建议"""
        recommendations = []

        # 从优化结果中提取建议
        if 'pricing' in optimizations and 'recommendations' in optimizations['pricing']:
            for rec in optimizations['pricing']['recommendations']:
                recommendations.append({
                    'type': 'pricing',
                    'priority': 'high' if abs(rec.get('expected_impact', 0)) > 1000 else 'medium',
                    'action': rec.get('action'),
                    'details': rec
                })

        # 从AI洞察中提取建议
        if 'problem_diagnosis' in insights and 'recommendations' in insights['problem_diagnosis']:
            for rec in insights['problem_diagnosis']['recommendations']:
                recommendations.append({
                    'type': 'operational',
                    'priority': rec.priority,
                    'action': rec.action,
                    'details': rec.dict()
                })

        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return recommendations[:10]  # 返回前10个建议

    def _format_metrics(self, data: Dict) -> str:
        """格式化指标数据"""
        return f"""
- 订单数：{data.get('order_count', 0)}
- 营收：¥{data.get('total_revenue', 0):,.0f}
- 客单价：¥{data.get('avg_order_value', 0):.0f}
- 客户数：{data.get('customer_count', 0)}
"""

    def _format_comparison(self, today: Dict, yesterday: Dict) -> str:
        """格式化对比数据"""
        revenue_change = (
                (today.get('total_revenue', 0) - yesterday.get('total_revenue', 0))
                / yesterday.get('total_revenue', 1) * 100
        )

        return f"""
- 营收变化：{revenue_change:+.1f}%
- 订单变化：{today.get('order_count', 0) - yesterday.get('order_count', 0):+d}
"""

    def _format_anomaly_factors(self, anomalies: Dict) -> str:
        """格式化异常因素"""
        factors = []

        if 'sales_anomalies' in anomalies:
            for anomaly in anomalies['sales_anomalies'][:3]:  # 前3个
                reasons = anomaly.get('anomaly_reasons', [])
                if reasons:
                    factors.extend(reasons)

        return '\n'.join(f"- {factor}" for factor in factors)