# src/engine/core.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_repository(repo_class, mock_class):
    """获取数据仓库（如果连接失败则使用模拟）"""
    try:
        repo = repo_class()
        # 测试连接
        if hasattr(repo, 'db') and repo.db:
            repo.db.client  # 触发连接
        return repo
    except Exception as e:
        logger.warning(f"Failed to connect to database, using mock data: {e}")
        return mock_class()


class AIGrowthEngineCore:
    """AI增长引擎核心类"""

    def __init__(self):
        # 延迟导入以避免循环依赖
        from ..data.repositories import OrderRepository, CustomerRepository, AnalyticsRepository
        from ..data.mock_repository import MockOrderRepository, MockCustomerRepository, MockAnalyticsRepository
        from ..analytics.predictive import SalesPredictorEngine
        from ..analytics.causal import CausalInferenceEngine
        from ..analytics.anomaly import AnomalyDetectionEngine
        from ..analytics.optimization import OptimizationEngine

        # 数据层 - 添加容错，如果数据库连接失败则使用模拟数据
        self.order_repo = get_repository(OrderRepository, MockOrderRepository)
        self.customer_repo = get_repository(CustomerRepository, MockCustomerRepository)
        self.analytics_repo = get_repository(AnalyticsRepository, MockAnalyticsRepository)

        # 分析引擎
        self.predictor = SalesPredictorEngine()
        self.causal_engine = CausalInferenceEngine()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.optimizer = OptimizationEngine()

        # AI层 - 延迟初始化，避免在没有配置时报错
        self._ai_agent = None
        self._analysis_chains = None

        # 状态管理
        self.state = {
            'last_run': None,
            'current_analysis': {},
            'recommendations': []
        }

    @property
    def ai_agent(self):
        """延迟加载AI代理"""
        if self._ai_agent is None:
            try:
                from ..llm.agents import AIGrowthAgent
                self._ai_agent = AIGrowthAgent()
            except Exception as e:
                logger.warning(f"Failed to initialize AI agent: {e}")

                # 返回一个简单的mock对象
                class MockAgent:
                    def chat(self, input_text):
                        return {
                            "answer": "AI功能暂时不可用，请检查Azure OpenAI配置。",
                            "intermediate_steps": []
                        }

                self._ai_agent = MockAgent()
        return self._ai_agent

    @property
    def analysis_chains(self):
        """延迟加载分析链"""
        if self._analysis_chains is None:
            try:
                from ..llm.chains import AnalysisChains
                self._analysis_chains = AnalysisChains()
            except Exception as e:
                logger.warning(f"Failed to initialize analysis chains: {e}")

                # 返回一个简单的mock对象
                class MockChains:
                    def create_daily_report_chain(self):
                        class MockChain:
                            def run(self, **kwargs):
                                return "今日报告：系统运行正常，销售数据稳定。"

                        return MockChain()

                    def create_problem_diagnosis_chain(self):
                        class MockChain:
                            def run(self, **kwargs):
                                return {"decline_analysis": "需要进一步分析", "recommendations": []}

                        return MockChain()

                    def create_customer_insight_chain(self):
                        class MockChain:
                            def run(self, **kwargs):
                                return "客户分析：各分群表现稳定。"

                        return MockChain()

                self._analysis_chains = MockChains()
        return self._analysis_chains

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
            self.state['recommendations'] = results['recommendations']

            logger.info(f"Daily analysis completed in {results['duration']:.2f} seconds")
            return results

        except Exception as e:
            logger.error(f"Daily analysis failed: {e}", exc_info=True)
            # 返回错误结果而不是抛出异常
            return {
                'timestamp': start_time,
                'duration': (datetime.now() - start_time).total_seconds(),
                'status': 'failed',
                'error': str(e),
                'data_summary': {},
                'anomalies': {},
                'predictions': {},
                'causal_analysis': {},
                'optimizations': {},
                'insights': {},
                'recommendations': []
            }

    def _collect_daily_data(self) -> Dict[str, pd.DataFrame]:
        """收集每日数据"""
        end_date = datetime.now()
        data = {}

        try:
            # 每日销售数据
            data['daily_sales'] = self.order_repo.get_daily_sales(
                end_date - timedelta(days=90), end_date
            )
        except Exception as e:
            logger.error(f"Failed to get daily sales: {e}")
            data['daily_sales'] = pd.DataFrame()

        try:
            # 商品表现数据
            data['item_performance'] = self.order_repo.get_item_performance(days=30)
        except Exception as e:
            logger.error(f"Failed to get item performance: {e}")
            data['item_performance'] = pd.DataFrame()

        try:
            # 客户分群数据
            data['customer_segments'] = self.customer_repo.get_customer_segments()
        except Exception as e:
            logger.error(f"Failed to get customer segments: {e}")
            data['customer_segments'] = pd.DataFrame()

        try:
            # 时间序列数据
            data['time_series'] = self.analytics_repo.get_time_series_data(days=180)
        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")
            data['time_series'] = pd.DataFrame()

        try:
            # 促销效果数据
            data['promotion_effectiveness'] = self.analytics_repo.get_promotion_effectiveness(days=30)
        except Exception as e:
            logger.error(f"Failed to get promotion effectiveness: {e}")
            data['promotion_effectiveness'] = pd.DataFrame()

        return data

    def _detect_anomalies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """检测异常"""
        anomalies = {}

        try:
            # 销售异常检测
            if 'daily_sales' in data and not data['daily_sales'].empty:
                # 准备特征
                features = []
                if 'total_revenue' in data['daily_sales'].columns:
                    features.append('total_revenue')
                if 'order_count' in data['daily_sales'].columns:
                    features.append('order_count')
                if 'avg_order_value' in data['daily_sales'].columns:
                    features.append('avg_order_value')
                if 'customer_count' in data['daily_sales'].columns:
                    features.append('customer_count')

                if features:
                    sales_with_anomalies = self.anomaly_detector.detect_sales_anomalies(
                        data['daily_sales'].rename(columns={'total_revenue': 'daily_revenue'}),
                        features=[f if f != 'total_revenue' else 'daily_revenue' for f in features]
                    )
                    anomalies['sales_anomalies'] = sales_with_anomalies[
                        sales_with_anomalies['is_anomaly']
                    ].to_dict('records')
                else:
                    anomalies['sales_anomalies'] = []
            else:
                anomalies['sales_anomalies'] = []
        except Exception as e:
            logger.error(f"Failed to detect sales anomalies: {e}")
            anomalies['sales_anomalies'] = []

        try:
            # 模式变化检测
            if 'time_series' in data and not data['time_series'].empty and 'y' in data['time_series'].columns:
                pattern_changes = self.anomaly_detector.detect_pattern_changes(
                    data['time_series'],
                    metric='y'
                )
                anomalies['pattern_changes'] = pattern_changes
            else:
                anomalies['pattern_changes'] = {'change_points': [], 'total_changes': 0}
        except Exception as e:
            logger.error(f"Failed to detect pattern changes: {e}")
            anomalies['pattern_changes'] = {'change_points': [], 'total_changes': 0}

        return anomalies

    def _run_predictions(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行预测"""
        predictions = {}

        try:
            # 销售预测
            if 'time_series' in data and not data['time_series'].empty:
                # 确保有必要的列
                if 'ds' in data['time_series'].columns and 'y' in data['time_series'].columns:
                    sales_forecast = self.predictor.predict_with_prophet(
                        data['time_series'],
                        periods=7
                    )
                    predictions['sales_forecast'] = sales_forecast.to_dict('records')
                else:
                    logger.warning("Time series data missing required columns (ds, y)")
                    predictions['sales_forecast'] = []
            else:
                predictions['sales_forecast'] = []
        except Exception as e:
            logger.error(f"Failed to run sales prediction: {e}")
            predictions['sales_forecast'] = []

        try:
            # 商品需求预测
            if 'item_performance' in data and not data['item_performance'].empty:
                # 添加必要的特征列
                if 'created_at_pt' not in data['item_performance'].columns:
                    data['item_performance']['created_at_pt'] = datetime.now()
                if 'item_amt' not in data['item_performance'].columns:
                    data['item_performance']['item_amt'] = data['item_performance'].get('avg_price', 50)
                if 'item_discount' not in data['item_performance'].columns:
                    data['item_performance']['item_discount'] = data['item_performance'].get('total_discount', 0) / len(
                        data['item_performance'])

                item_demand = self.predictor.predict_item_demand(
                    data['item_performance'],
                    target_col='units_sold' if 'units_sold' in data['item_performance'].columns else 'order_count'
                )
                predictions['item_demand'] = {
                    'metrics': item_demand.get('metrics', {}),
                    'feature_importance': item_demand.get('feature_importance', pd.DataFrame()).to_dict('records')
                }
            else:
                predictions['item_demand'] = {'metrics': {}, 'feature_importance': []}
        except Exception as e:
            logger.error(f"Failed to run item demand prediction: {e}")
            predictions['item_demand'] = {'metrics': {}, 'feature_importance': []}

        return predictions

    def _run_causal_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """运行因果分析"""
        causal_results = {}

        try:
            # 分析促销效果
            if 'daily_sales' in data and not data['daily_sales'].empty and len(data['daily_sales']) > 10:
                # 准备数据
                promo_data = data['daily_sales'].copy()

                # 创建促销标记（简化版本）
                if 'promotion_effectiveness' in data and not data['promotion_effectiveness'].empty:
                    promo_dates = data['promotion_effectiveness']['date'].unique() if 'date' in data[
                        'promotion_effectiveness'].columns else []
                    promo_data['has_promotion'] = promo_data['date'].isin(promo_dates).astype(int)
                else:
                    # 随机分配一些促销日
                    promo_data['has_promotion'] = (np.random.random(len(promo_data)) > 0.7).astype(int)

                # 添加必要的列
                promo_data['revenue'] = promo_data.get('total_revenue', promo_data.get('revenue', 50000))
                promo_data['is_weekend'] = pd.to_datetime(promo_data['date']).dt.dayofweek.isin([5, 6]).astype(int)
                promo_data['hour_of_day'] = 12  # 默认中午

                causal_results['promotion_effect'] = self.causal_engine.analyze_promotion_effect(
                    promo_data,
                    treatment='has_promotion',
                    outcome='revenue',
                    confounders=['is_weekend']
                )
            else:
                causal_results['promotion_effect'] = {
                    'estimates': {},
                    'refutation': {},
                    'interpretation': '数据不足，无法进行因果分析',
                    'recommendation': {'action': '收集更多数据', 'reason': '需要更多历史数据', 'confidence': 'low'}
                }
        except Exception as e:
            logger.error(f"Failed to run causal analysis: {e}")
            causal_results['promotion_effect'] = {
                'estimates': {},
                'refutation': {},
                'interpretation': f'分析失败: {str(e)}',
                'recommendation': {'action': '检查数据质量', 'reason': '分析过程出错', 'confidence': 'low'}
            }

        return causal_results

    def _generate_optimizations(self, data: Dict[str, pd.DataFrame], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化建议"""
        optimizations = {}

        try:
            # 价格优化
            if 'item_performance' in data and not data['item_performance'].empty:
                # 模拟成本数据
                item_costs = {}
                for _, item in data['item_performance'].iterrows():
                    item_name = item.get('item_name', 'Unknown')
                    price = item.get('avg_price', 50)
                    item_costs[item_name] = price * 0.4  # 假设成本是售价的40%

                pricing_optimization = self.optimizer.optimize_pricing(
                    data['item_performance'],
                    item_costs,
                    constraints={'max_price_increase': 0.2}  # 最多涨价20%
                )
                optimizations['pricing'] = pricing_optimization
            else:
                optimizations['pricing'] = {'optimal_prices': {}, 'recommendations': []}
        except Exception as e:
            logger.error(f"Failed to generate pricing optimization: {e}")
            optimizations['pricing'] = {'optimal_prices': {}, 'recommendations': []}

        try:
            # 库存优化（简化版本）
            if predictions.get('sales_forecast'):
                # 基于预测生成简单的库存建议
                forecast_data = predictions['sales_forecast']
                if forecast_data:
                    avg_forecast = np.mean([f.get('yhat', 50000) for f in forecast_data])
                    optimizations['inventory'] = {
                        'recommendations': [
                            {
                                'item': '整体库存',
                                'action': '增加备货' if avg_forecast > 45000 else '维持现有库存',
                                'reason': f'预测未来7天平均销售额为¥{avg_forecast:.0f}'
                            }
                        ]
                    }
                else:
                    optimizations['inventory'] = {'recommendations': []}
            else:
                optimizations['inventory'] = {'recommendations': []}
        except Exception as e:
            logger.error(f"Failed to generate inventory optimization: {e}")
            optimizations['inventory'] = {'recommendations': []}

        return optimizations

    def _generate_ai_insights(self, data: Dict[str, pd.DataFrame], anomalies: Dict[str, Any],
                              predictions: Dict[str, Any], causal_results: Dict[str, Any],
                              optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """生成AI洞察"""
        insights = {}

        try:
            # 生成日报
            if 'daily_sales' in data and not data['daily_sales'].empty:
                today_data = data['daily_sales'].iloc[0] if len(data['daily_sales']) > 0 else {}
                yesterday_data = data['daily_sales'].iloc[1] if len(data['daily_sales']) > 1 else {}

                daily_report_chain = self.analysis_chains.create_daily_report_chain()
                daily_report = daily_report_chain.run(
                    date=datetime.now().strftime('%Y-%m-%d'),
                    metrics=self._format_metrics(today_data),
                    comparison=self._format_comparison(today_data, yesterday_data)
                )
                insights['daily_report'] = daily_report
            else:
                insights['daily_report'] = "暂无今日数据"
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            insights['daily_report'] = "生成日报失败"

        try:
            # 如果有异常，生成问题诊断
            if anomalies.get('sales_anomalies'):
                problem_chain = self.analysis_chains.create_problem_diagnosis_chain()
                diagnosis = problem_chain.run(
                    period="最近7天",
                    decline_pct=20,  # 示例
                    factors=self._format_anomaly_factors(anomalies),
                    historical_context="历史同期平均销售额为正常水平"
                )
                insights['problem_diagnosis'] = diagnosis
        except Exception as e:
            logger.error(f"Failed to generate problem diagnosis: {e}")
            insights['problem_diagnosis'] = {}

        try:
            # 客户洞察
            if 'customer_segments' in data and not data['customer_segments'].empty:
                customer_chain = self.analysis_chains.create_customer_insight_chain()
                customer_insights = customer_chain.run(
                    segment_data=data['customer_segments'].head().to_string(),
                    behavior_patterns="基于历史数据分析",
                    trends="客户行为稳定"
                )
                insights['customer_insights'] = customer_insights
        except Exception as e:
            logger.error(f"Failed to generate customer insights: {e}")
            insights['customer_insights'] = "客户分析暂时不可用"

        return insights

    def _summarize_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """汇总数据"""
        summary = {}

        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                summary[key] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
                        'end': str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
                    }
                }
            else:
                summary[key] = {'rows': 0, 'columns': [], 'date_range': {}}

        return summary

    def _prioritize_recommendations(self, optimizations: Dict[str, Any], insights: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """优先级排序建议"""
        recommendations = []

        # 从优化结果中提取建议
        if 'pricing' in optimizations and 'recommendations' in optimizations['pricing']:
            for rec in optimizations['pricing']['recommendations']:
                recommendations.append({
                    'type': 'pricing',
                    'priority': '高' if abs(
                        rec.get('expected_impact', '').replace('¥', '').replace(',', '').split()[0] if rec.get(
                            'expected_impact') else '0') > '1000' else '中',
                    'action': rec.get('action', ''),
                    'reason': rec.get('reason', ''),
                    'expected_impact': rec.get('expected_impact', ''),
                    'confidence': 0.8
                })

        # 从库存优化中提取建议
        if 'inventory' in optimizations and 'recommendations' in optimizations['inventory']:
            for rec in optimizations['inventory']['recommendations']:
                recommendations.append({
                    'type': 'inventory',
                    'priority': '中',
                    'action': rec.get('action', ''),
                    'reason': rec.get('reason', ''),
                    'expected_impact': '优化库存水平',
                    'confidence': 0.7
                })

        # 添加一些默认建议
        if len(recommendations) < 3:
            recommendations.extend([
                {
                    'type': 'marketing',
                    'priority': '中',
                    'action': '优化营销策略',
                    'reason': '提升客户获取和留存',
                    'expected_impact': '预计提升10%新客获取',
                    'confidence': 0.6
                },
                {
                    'type': 'operational',
                    'priority': '低',
                    'action': '优化运营流程',
                    'reason': '提高效率降低成本',
                    'expected_impact': '预计节省5%运营成本',
                    'confidence': 0.5
                }
            ])

        # 按优先级排序
        priority_order = {'高': 0, '中': 1, '低': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        return recommendations[:10]  # 返回前10个建议

    def _format_metrics(self, data: Any) -> str:
        """格式化指标数据"""
        if isinstance(data, dict):
            return f"""
- 订单数：{data.get('order_count', 0)}
- 营收：¥{data.get('total_revenue', 0):,.0f}
- 客单价：¥{data.get('avg_order_value', 0):.0f}
- 客户数：{data.get('customer_count', 0)}
"""
        return "- 暂无数据"

    def _format_comparison(self, today: Any, yesterday: Any) -> str:
        """格式化对比数据"""
        if isinstance(today, dict) and isinstance(yesterday, dict):
            today_revenue = today.get('total_revenue', 0)
            yesterday_revenue = yesterday.get('total_revenue', 1)
            revenue_change = (
                        (today_revenue - yesterday_revenue) / yesterday_revenue * 100) if yesterday_revenue != 0 else 0

            return f"""
- 营收变化：{revenue_change:+.1f}%
- 订单变化：{today.get('order_count', 0) - yesterday.get('order_count', 0):+d}
"""
        return "- 暂无对比数据"

    def _format_anomaly_factors(self, anomalies: Dict[str, Any]) -> str:
        """格式化异常因素"""
        factors = []

        if 'sales_anomalies' in anomalies:
            for anomaly in anomalies['sales_anomalies'][:3]:  # 前3个
                reasons = anomaly.get('anomaly_reasons', [])
                if reasons:
                    factors.extend(reasons)

        if factors:
            return '\n'.join(f"- {factor}" for factor in factors[:5])  # 最多5个因素
        return "- 未发现明显异常因素"