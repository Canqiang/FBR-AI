import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class SalesPredictorEngine:
    """销售预测引擎"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def predict_with_prophet(
            self,
            data: pd.DataFrame,
            periods: int = 7,
            include_holidays: bool = True
    ) -> pd.DataFrame:
        """使用Prophet进行时间序列预测"""
        logger.info(f"Starting Prophet prediction for {periods} periods")

        # 确保数据格式正确
        if 'ds' not in data.columns or 'y' not in data.columns:
            raise ValueError("Data must have 'ds' and 'y' columns for Prophet")

        # 初始化Prophet模型
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )

        # 添加中国节假日
        if include_holidays:
            model.add_country_holidays(country_name='CN')

        # 添加额外的回归变量（如果有）
        extra_regressors = [col for col in data.columns if col not in ['ds', 'y']]
        for regressor in extra_regressors:
            model.add_regressor(regressor)

        data = data.dropna()
        # 训练模型
        model.fit(data)

        # 创建未来日期框架
        future = model.make_future_dataframe(periods=periods)

        # 如果有额外回归变量，需要为未来日期提供值
        for regressor in extra_regressors:
            # 使用历史平均值作为未来值的估计
            future[regressor] = data[regressor].mean()

        # 进行预测
        forecast = model.predict(future)

        # 提取关键预测结果
        prediction_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly', 'yearly']
        result = forecast[prediction_cols].tail(periods)

        # 计算预测质量指标
        if len(data) > 30:  # 有足够的历史数据进行验证
            cv_results = self._cross_validate_prophet(model, data)
            logger.info(f"Prophet CV MAPE: {cv_results['mape']:.2%}")

        self.models['prophet'] = model

        return result

    def _cross_validate_prophet(self, model: Prophet, data: pd.DataFrame) -> Dict[str, float]:
        """Prophet模型交叉验证"""
        from prophet.diagnostics import cross_validation, performance_metrics

        # 设置交叉验证参数
        initial = f'{len(data) * 0.7:.0f} days'
        period = '7 days'
        horizon = '7 days'

        try:
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            df_p = performance_metrics(df_cv)

            return {
                'mape': df_p['mape'].mean(),
                'rmse': df_p['rmse'].mean(),
                'mae': df_p['mae'].mean()
            }
        except Exception as e:
            logger.warning(f"Cross validation failed: {e}")
            return {'mape': 0, 'rmse': 0, 'mae': 0}

    def predict_item_demand(
            self,
            data: pd.DataFrame,
            target_col: str = 'units_sold',
            feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """使用XGBoost预测商品需求"""
        logger.info("Starting XGBoost demand prediction")

        # 特征工程
        if feature_cols is None:
            feature_cols = [
                'day_of_week', 'is_weekend', 'is_holiday',
                'avg_price', 'discount_rate', 'category_encoded'
            ]

        # 准备特征
        X = self._prepare_features(data, feature_cols)
        y = data[target_col]

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )

        model.fit(X_train, y_train)

        # 预测和评估
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 特征重要性
        feature_importance = pd.DataFrame(list(zip(feature_cols, model.feature_importances_)), columns=["feature", "importance"]).sort_values('importance', ascending=False)

        # feature_importance = pd.DataFrame({
        #     'feature': feature_cols,
        #     'importance': model.feature_importances_
        # }).sort_values('importance', ascending=False)

        self.models['xgboost_demand'] = model
        self.feature_importance['demand'] = feature_importance

        return {
            'model': model,
            'metrics': {'mae': mae, 'rmse': rmse},
            'feature_importance': feature_importance,
            'predictions': y_pred
        }

    def _prepare_features(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """准备模型特征"""
        features = pd.DataFrame()

        # 时间特征
        if 'created_at_pt' in data.columns:
            features['day_of_week'] = pd.to_datetime(data['created_at_pt']).dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['hour'] = pd.to_datetime(data['created_at_pt']).dt.hour

        # 价格特征
        if 'item_amt' in data.columns and 'item_discount' in data.columns:
            features['avg_price'] = data['item_amt']
            features['discount_rate'] = data['item_discount'] / (data['item_amt'] + 0.01)

        # 类别编码
        if 'category_name' in data.columns:
            features['category_encoded'] = pd.Categorical(data['category_name']).codes

        # 只返回需要的特征
        available_features = [col for col in feature_cols if col in features.columns]
        return features[available_features]