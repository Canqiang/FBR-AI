import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

logger = logging.getLogger(__name__)


class AnomalyDetectionEngine:
    """异常检测引擎"""

    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.thresholds = {}

    def detect_sales_anomalies(
            self,
            data: pd.DataFrame,
            features: Optional[List[str]] = None,
            contamination: float = 0.1
    ) -> pd.DataFrame:
        """检测销售数据异常"""
        logger.info("Detecting sales anomalies")

        # 默认特征
        if features is None:
            features = ['daily_revenue', 'order_count', 'avg_order_value', 'customer_count']

        # 过滤可用特征
        available_features = [f for f in features if f in data.columns]

        if not available_features:
            logger.warning("No features available for anomaly detection")
            return data

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[available_features])
        self.scalers['sales'] = scaler

        # 使用Isolation Forest进行异常检测
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # 预测异常
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)

        # 添加异常标记和分数
        data['is_anomaly'] = anomaly_labels == -1
        data['anomaly_score'] = anomaly_scores

        # 分析异常原因
        data['anomaly_reasons'] = data.apply(
            lambda row: self._analyze_anomaly_reasons(row, available_features, data),
            axis=1
        )

        self.models['sales_isolation_forest'] = iso_forest

        return data

    def _analyze_anomaly_reasons(
            self,
            row: pd.Series,
            features: List[str],
            data: pd.DataFrame
    ) -> List[str]:
        """分析异常原因"""
        if not row.get('is_anomaly', False):
            return []

        reasons = []

        for feature in features:
            if feature not in row:
                continue

            # 计算该特征的统计信息
            mean = data[feature].mean()
            std = data[feature].std()
            value = row[feature]

            # Z-score
            z_score = (value - mean) / std if std > 0 else 0

            # 判断异常类型
            if abs(z_score) > 2:
                if z_score > 0:
                    reasons.append(f"{feature}异常高（{value:.0f}，平均值{mean:.0f}）")
                else:
                    reasons.append(f"{feature}异常低（{value:.0f}，平均值{mean:.0f}）")

        return reasons

    def detect_pattern_changes(
            self,
            data: pd.DataFrame,
            metric: str,
            window_size: int = 7
    ) -> Dict[str, Any]:
        """检测模式变化（如趋势突变）"""
        logger.info(f"Detecting pattern changes in {metric}")

        if metric not in data.columns:
            raise ValueError(f"Metric {metric} not found in data")

        # 计算移动平均和标准差
        data[f'{metric}_ma'] = data[metric].rolling(window=window_size).mean()
        data[f'{metric}_std'] = data[metric].rolling(window=window_size).std()

        # 检测突变点
        change_points = []

        for i in range(window_size * 2, len(data)):
            # 比较前后两个窗口
            before = data[metric].iloc[i - window_size * 2:i - window_size]
            after = data[metric].iloc[i - window_size:i]

            # 使用t检验检测均值变化
            t_stat, p_value = stats.ttest_ind(before, after)

            if p_value < 0.05:  # 显著性水平
                change_magnitude = after.mean() - before.mean()
                change_pct = change_magnitude / before.mean() * 100 if before.mean() != 0 else 0

                change_points.append({
                    'index': i,
                    'date': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i,
                    'change_magnitude': change_magnitude,
                    'change_pct': change_pct,
                    'p_value': p_value,
                    'before_mean': before.mean(),
                    'after_mean': after.mean()
                })

        return {
            'change_points': change_points,
            'total_changes': len(change_points),
            'significant_changes': [cp for cp in change_points if abs(cp['change_pct']) > 20]
        }

    def detect_multivariate_outliers(
            self,
            data: pd.DataFrame,
            features: List[str],
            method: str = 'mahalanobis'
    ) -> pd.DataFrame:
        """检测多变量异常值"""
        logger.info(f"Detecting multivariate outliers using {method}")

        if method == 'mahalanobis':
            # 计算Mahalanobis距离
            data['outlier_score'] = self._calculate_mahalanobis_distance(data[features])

            # 使用卡方分布确定阈值
            threshold = stats.chi2.ppf(0.95, df=len(features))
            data['is_outlier'] = data['outlier_score'] > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        return data

    def _calculate_mahalanobis_distance(self, data: pd.DataFrame) -> pd.Series:
        """计算Mahalanobis距离"""
        # 计算均值和协方差矩阵
        mean = data.mean()
        cov = data.cov()

        # 计算协方差矩阵的逆
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            inv_cov = np.linalg.pinv(cov)

        # 计算每个点的Mahalanobis距离
        distances = []
        for _, row in data.iterrows():
            diff = row - mean
            distance = np.sqrt(diff.values @ inv_cov @ diff.values)
            distances.append(distance)

        return pd.Series(distances, index=data.index)