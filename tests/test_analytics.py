import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analytics.predictive import SalesPredictorEngine
from src.analytics.causal import CausalInferenceEngine
from src.analytics.anomaly import AnomalyDetectionEngine


class TestSalesPredictorEngine:
    """测试销售预测引擎"""

    @pytest.fixture
    def predictor(self):
        """创建预测器实例"""
        return SalesPredictorEngine()

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(30000, 60000, size=90) + np.sin(np.arange(90) * 0.1) * 5000
        })
        return data

    def test_prophet_prediction(self, predictor, sample_data):
        """测试Prophet预测"""
        # 运行预测
        forecast = predictor.predict_with_prophet(sample_data, periods=7)

        # 验证结果
        assert len(forecast) == 7
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns

        # 验证预测值合理性
        assert forecast['yhat'].min() > 0
        assert forecast['yhat_lower'].min() < forecast['yhat'].min()
        assert forecast['yhat_upper'].max() > forecast['yhat'].max()

    def test_item_demand_prediction(self, predictor):
        """测试商品需求预测"""
        # 创建商品数据
        data = pd.DataFrame({
            'item_name': ['奶茶'] * 30 + ['咖啡'] * 30,
            'units_sold': np.random.randint(50, 150, size=60),
            'created_at_pt': pd.date_range(end=datetime.now(), periods=60),
            'item_amt': [15] * 30 + [20] * 30,
            'item_discount': np.random.randint(0, 5, size=60),
            'category_name': ['饮品'] * 60
        })

        # 运行预测
        result = predictor.predict_item_demand(data)

        # 验证结果
        assert 'model' in result
        assert 'metrics' in result
        assert 'feature_importance' in result
        assert result['metrics']['mae'] >= 0
        assert len(result['feature_importance']) > 0


class TestCausalInferenceEngine:
    """测试因果推断引擎"""

    @pytest.fixture
    def causal_engine(self):
        """创建因果推断引擎实例"""
        return CausalInferenceEngine()

    @pytest.fixture
    def promotion_data(self):
        """创建促销数据"""
        n_samples = 1000

        # 创建混淆因素
        is_weekend = np.random.binomial(1, 0.3, n_samples)
        hour_of_day = np.random.randint(10, 22, n_samples)

        # 处理变量（促销）- 受混淆因素影响
        has_promotion = np.random.binomial(
            1,
            0.3 + 0.2 * is_weekend + 0.01 * (hour_of_day - 16)
        )

        # 结果变量（营收）- 受处理和混淆因素影响
        revenue = (
                1000 +
                200 * has_promotion +  # 真实促销效果
                300 * is_weekend +  # 周末效应
                20 * hour_of_day +  # 时段效应
                np.random.normal(0, 100, n_samples)  # 噪声
        )

        return pd.DataFrame({
            'has_promotion': has_promotion,
            'revenue': revenue,
            'is_weekend': is_weekend,
            'hour_of_day': hour_of_day
        })

    def test_promotion_effect_analysis(self, causal_engine, promotion_data):
        """测试促销效果分析"""
        # 运行因果分析
        result = causal_engine.analyze_promotion_effect(
            promotion_data,
            treatment='has_promotion',
            outcome='revenue',
            confounders=['is_weekend', 'hour_of_day']
        )

        # 验证结果结构
        assert 'estimates' in result
        assert 'refutation' in result
        assert 'interpretation' in result
        assert 'recommendation' in result

        # 验证估计值合理性（应该接近200）
        estimates = result['estimates']
        if 'linear_regression' in estimates:
            effect = estimates['linear_regression']['effect']
            assert 150 < effect < 250  # 真实效果是200


class TestAnomalyDetectionEngine:
    """测试异常检测引擎"""

    @pytest.fixture
    def anomaly_detector(self):
        """创建异常检测引擎实例"""
        return AnomalyDetectionEngine()

    @pytest.fixture
    def sales_data_with_anomalies(self):
        """创建包含异常的销售数据"""
        # 正常数据
        normal_data = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=100),
            'daily_revenue': np.random.normal(50000, 5000, 100),
            'order_count': np.random.normal(800, 100, 100),
            'avg_order_value': np.random.normal(60, 5, 100),
            'customer_count': np.random.normal(600, 80, 100)
        })

        # 注入异常
        anomaly_indices = [20, 50, 80]
        for idx in anomaly_indices:
            normal_data.loc[idx, 'daily_revenue'] = 20000  # 异常低
            normal_data.loc[idx, 'order_count'] = 300

        return normal_data

    def test_sales_anomaly_detection(self, anomaly_detector, sales_data_with_anomalies):
        """测试销售异常检测"""
        # 运行异常检测
        result = anomaly_detector.detect_sales_anomalies(
            sales_data_with_anomalies,
            contamination=0.1
        )

        # 验证结果
        assert 'is_anomaly' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'anomaly_reasons' in result.columns

        # 验证检测到异常
        anomalies = result[result['is_anomaly']]
        assert len(anomalies) > 0
        assert len(anomalies) <= len(sales_data_with_anomalies) * 0.1  # 不超过10%

        # 验证异常原因
        for _, anomaly in anomalies.iterrows():
            assert len(anomaly['anomaly_reasons']) > 0