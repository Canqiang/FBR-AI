import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple
import warnings
from .extra_features import create_holiday_features,create_weather_features
warnings.filterwarnings('ignore')
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class FBRUSCausalAnalyzer:
    """FBR美国门店数据的因果分析器"""

    def __init__(self, store_locations: List[Dict[str, str]]):
        """
        初始化分析器

        Args:
            store_locations: 门店信息列表 [{'name': '门店名', 'postal_code': '邮编'}, ...]
        """
        self.store_locations = store_locations
        self.sales_data = None
        self.weather_data = None
        self.holiday_data = None
        self.merged_data = None

    def load_sales_data(self, filepath: str = None) -> pd.DataFrame:
        """加载销售数据"""
        logger.info("加载销售数据...")

        # 如果没有提供文件路径，生成模拟数据
        if filepath is None:
            logger.info("使用模拟数据...")
            return self._generate_mock_sales_data()

        # 实际加载逻辑
        sales_data = pd.read_csv(filepath)
        self.sales_data = sales_data
        return sales_data

    def _generate_mock_sales_data(self) -> pd.DataFrame:
        """生成模拟的销售数据"""
        np.random.seed(42)

        # 生成日期范围
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 生成多个门店的数据
        all_data = []

        for store in self.store_locations:
            n_days = len(dates)

            # 基础销售额（有季节性）
            base_sales = 50000 + 10000 * np.sin(np.arange(n_days) * 2 * np.pi / 365)

            # 添加随机波动
            daily_variation = np.random.normal(0, 5000, n_days)

            # 周末效应
            weekend_effect = np.array([1.2 if d.weekday() >= 5 else 1.0 for d in dates])

            # 促销效应（随机20%的日子有促销）
            promotion = np.random.binomial(1, 0.2, n_days)
            promotion_effect = 1 + promotion * 0.15

            # 计算总销售额
            sales = base_sales * weekend_effect * promotion_effect + daily_variation

            # 创建数据框
            location_data = pd.DataFrame({
                'date': dates,
                'store_name': store['name'],
                'postal_code': store['postal_code'],
                'sales_revenue': sales,
                'orders_count': sales / 80 + np.random.normal(0, 20, n_days),  # 平均客单价80
                'has_promotion': promotion,
                'discount_rate': promotion * np.random.uniform(0.1, 0.3, n_days),
                'inventory_level': np.random.uniform(0.3, 1.0, n_days),
                'day_of_week': [d.weekday() for d in dates],
                'is_weekend': [d.weekday() >= 5 for d in dates]
            })

            all_data.append(location_data)

        self.sales_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"生成了 {len(self.sales_data)} 条销售记录")
        return self.sales_data

    def fetch_weather_data(self) -> pd.DataFrame:
        """获取天气数据"""
        logger.info("获取天气数据...")

        # 准备邮编和日期范围
        postal_code_date_ranges = {}

        for store in self.store_locations:
            # 获取该店铺数据的日期范围
            store_data = self.sales_data[self.sales_data['postal_code'] == store['postal_code']]
            if not store_data.empty:
                postal_code_date_ranges[store['postal_code']] = {
                    'start_date': store_data['date'].min().strftime('%Y-%m-%d'),
                    'end_date': store_data['date'].max().strftime('%Y-%m-%d')
                }

        # 调用API获取天气数据
        self.weather_data = create_weather_features(postal_code_date_ranges)

        # 添加衍生特征
        if not self.weather_data.empty:
            self.weather_data['temperature_avg'] = (
                self.weather_data['temperature_2m_max'] + self.weather_data['temperature_2m_min']
            ) / 2

            # 将摄氏度转换为华氏度（美国常用）
            self.weather_data['temp_max_f'] = self.weather_data['temperature_2m_max'] * 9/5 + 32
            self.weather_data['temp_min_f'] = self.weather_data['temperature_2m_min'] * 9/5 + 32
            self.weather_data['temp_avg_f'] = self.weather_data['temperature_avg'] * 9/5 + 32

            # 天气分类
            self.weather_data['is_rainy'] = (self.weather_data['precipitation_sum'] > 5).astype(int)
            self.weather_data['is_snowy'] = (self.weather_data['snowfall_sum'] > 0).astype(int)
            self.weather_data['is_extreme_temp'] = (
                (self.weather_data['temp_max_f'] > 95) |  # 高于95°F
                (self.weather_data['temp_min_f'] < 32)    # 低于32°F（冰点）
            ).astype(int)
            self.weather_data['is_windy'] = (self.weather_data['wind_speed_10m_max'] > 30).astype(int)

            # 综合极端天气
            self.weather_data['is_extreme_weather'] = (
                self.weather_data['is_extreme_temp'] |
                self.weather_data['is_windy'] |
                (self.weather_data['precipitation_sum'] > 50) |
                (self.weather_data['snowfall_sum'] > 10)
            ).astype(int)

        logger.info(f"获取了 {len(self.weather_data)} 条天气记录")
        return self.weather_data

    def fetch_holiday_data(self) -> pd.DataFrame:
        """获取节假日数据"""
        logger.info("获取节假日数据...")

        # 准备邮编和日期范围
        postal_code_date_ranges = {}

        for store in self.store_locations:
            store_data = self.sales_data[self.sales_data['postal_code'] == store['postal_code']]
            if not store_data.empty:
                postal_code_date_ranges[store['postal_code']] = {
                    'start_date': store_data['date'].min().strftime('%Y-%m-%d'),
                    'end_date': store_data['date'].max().strftime('%Y-%m-%d')
                }

        # 调用API获取节假日数据
        self.holiday_data = create_holiday_features(postal_code_date_ranges)

        # 添加节假日分类
        if not self.holiday_data.empty:
            # 主要节假日分类
            major_holidays = [
                "New Year's Day", "Martin Luther King Jr. Day", "Presidents' Day",
                "Memorial Day", "Independence Day", "Labor Day",
                "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas Day"
            ]

            self.holiday_data['is_major_holiday'] = self.holiday_data['holiday_name'].isin(major_holidays).astype(int)

            # 购物季节标记
            self.holiday_data['is_shopping_season'] = (
                self.holiday_data['holiday_name'].isin(['Thanksgiving', 'Christmas Day']) |
                (self.holiday_data['days_until_next_holiday'] <= 7) &
                self.holiday_data['holiday_name'].str.contains('Christmas|Thanksgiving', na=False)
            ).astype(int)

        logger.info(f"获取了 {len(self.holiday_data)} 条节假日记录")
        return self.holiday_data

    def merge_all_data(self) -> pd.DataFrame:
        """合并所有数据源"""
        logger.info("合并数据...")

        # 确保日期格式一致
        self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])

        if not self.weather_data.empty:
            self.weather_data['date'] = pd.to_datetime(self.weather_data['date'])
            # 合并销售和天气数据
            merged = pd.merge(
                self.sales_data,
                self.weather_data,
                on=['date', 'postal_code'],
                how='left'
            )
        else:
            merged = self.sales_data.copy()
            # 添加默认天气列
            merged['temp_avg_f'] = 70
            merged['is_rainy'] = 0
            merged['is_extreme_weather'] = 0

        if not self.holiday_data.empty:
            self.holiday_data['date'] = pd.to_datetime(self.holiday_data['date'])
            # 合并节假日数据
            merged = pd.merge(
                merged,
                self.holiday_data,
                on=['date', 'postal_code'],
                how='left',
                suffixes=('', '_holiday')
            )
        else:
            # 添加默认节假日列
            merged['is_holiday'] = 0
            merged['holiday_name'] = "No Holiday"
            merged['days_until_next_holiday'] = 999

        # 填充缺失值
        merged = merged.fillna({
            'temp_avg_f': 70,
            'precipitation_sum': 0,
            'is_rainy': 0,
            'is_extreme_weather': 0,
            'is_holiday': 0,
            'holiday_name': "No Holiday",
            'days_until_next_holiday': 999
        })

        # 清理重复的state列
        if 'state_holiday' in merged.columns:
            merged['state'] = merged['state'].fillna(merged['state_holiday'])
            merged = merged.drop(columns=['state_holiday'])

        self.merged_data = merged
        logger.info(f"合并后数据集包含 {len(merged)} 条记录，{merged.shape[1]} 个特征")
        return merged

    def perform_causal_analysis(self) -> Dict[str, Any]:
        """执行因果分析"""
        logger.info("开始因果分析...")

        if self.merged_data is None:
            raise ValueError("请先合并数据")

        # 1. 天气对销售的因果效应
        weather_effect = self._analyze_weather_effect()

        # 2. 促销对销售的因果效应
        promotion_effect = self._analyze_promotion_effect()

        # 3. 节假日对销售的因果效应
        holiday_effect = self._analyze_holiday_effect()

        # 4. 综合因果模型
        combined_effect = self._analyze_combined_effects()

        return {
            'weather_effect': weather_effect,
            'promotion_effect': promotion_effect,
            'holiday_effect': holiday_effect,
            'combined_effect': combined_effect
        }

    def _analyze_weather_effect(self) -> Dict[str, Any]:
        """分析天气对销售的因果效应"""
        logger.info("分析天气因果效应...")

        # 准备数据
        data = self.merged_data.copy()

        # 只分析有足够数据的情况
        if 'is_rainy' not in data.columns or data['is_rainy'].sum() < 10:
            logger.warning("雨天数据不足，跳过雨天分析")
            return {'error': '数据不足'}

        try:
            # 创建因果模型 - 雨天效应
            model = CausalModel(
                data=data,
                treatment='is_rainy',
                outcome='sales_revenue',
                common_causes=['day_of_week', 'has_promotion', 'is_holiday']
            )

            # 识别因果效应
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # 估计效应
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate"
            )

            rainy_effect = {
                'effect': estimate.value,
                'interpretation': f"雨天导致销售额平均{'下降' if estimate.value < 0 else '上升'} ${abs(estimate.value):.0f}"
            }

        except Exception as e:
            logger.error(f"雨天效应分析失败: {e}")
            rainy_effect = {'error': str(e)}

        # 温度效应分析
        try:
            # 创建温度区间
            data['temp_category'] = pd.cut(
                data['temp_avg_f'],
                bins=[0, 32, 50, 70, 85, 100, 120],
                labels=['极冷(<32°F)', '冷(32-50°F)', '舒适(50-70°F)',
                       '温暖(70-85°F)', '热(85-100°F)', '极热(>100°F)']
            )

            temp_effects = data.groupby('temp_category')['sales_revenue'].agg(['mean', 'count'])
            optimal_temp = temp_effects['mean'].idxmax()

            temperature_effect = {
                'optimal_temperature': optimal_temp,
                'temperature_impact': temp_effects.to_dict(),
                'interpretation': f"最佳销售温度区间是{optimal_temp}"
            }

        except Exception as e:
            logger.error(f"温度效应分析失败: {e}")
            temperature_effect = {'error': str(e)}

        return {
            'rainy_day_effect': rainy_effect,
            'temperature_effect': temperature_effect,
            'extreme_weather_effect': self._analyze_extreme_weather_effect(data)
        }

    def _analyze_extreme_weather_effect(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析极端天气效应"""
        try:
            if 'is_extreme_weather' in data.columns and data['is_extreme_weather'].sum() > 5:
                extreme_days = data[data['is_extreme_weather'] == 1]['sales_revenue'].mean()
                normal_days = data[data['is_extreme_weather'] == 0]['sales_revenue'].mean()
                effect = extreme_days - normal_days

                return {
                    'effect': effect,
                    'extreme_days_count': data['is_extreme_weather'].sum(),
                    'interpretation': f"极端天气导致销售额平均{'下降' if effect < 0 else '上升'} ${abs(effect):.0f}"
                }
            else:
                return {'error': '极端天气数据不足'}
        except Exception as e:
            logger.error(f"极端天气分析失败: {e}")
            return {'error': str(e)}

    def _analyze_promotion_effect(self) -> Dict[str, Any]:
        """分析促销的因果效应"""
        logger.info("分析促销因果效应...")

        data = self.merged_data.copy()

        try:
            # 创建因果模型
            model = CausalModel(
                data=data,
                treatment='has_promotion',
                outcome='sales_revenue',
                common_causes=['day_of_week', 'is_holiday', 'is_rainy']
            )

            identified_estimand = model.identify_effect()

            # 估计平均处理效应
            ate_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate"
            )

            # 计算ROI
            avg_discount = data[data['has_promotion'] == 1]['discount_rate'].mean()
            avg_sales_with_promo = data[data['has_promotion'] == 1]['sales_revenue'].mean()
            discount_cost = avg_discount * avg_sales_with_promo
            roi = (ate_estimate.value - discount_cost) / discount_cost if discount_cost > 0 else 0

            # 分析异质性效应
            heterogeneous_effects = self._analyze_heterogeneous_effects(data, 'has_promotion', 'sales_revenue')

            return {
                'average_effect': ate_estimate.value,
                'roi': roi,
                'heterogeneous_effects': heterogeneous_effects,
                'interpretation': f"促销平均提升销售额 ${ate_estimate.value:.0f}，ROI约为 {roi:.1%}"
            }

        except Exception as e:
            logger.error(f"促销效应分析失败: {e}")
            return {'error': str(e)}

    def _analyze_holiday_effect(self) -> Dict[str, Any]:
        """分析节假日的因果效应"""
        logger.info("分析节假日因果效应...")

        data = self.merged_data.copy()

        try:
            # 整体节假日效应
            holiday_sales = data[data['is_holiday'] == 1]['sales_revenue'].mean()
            non_holiday_sales = data[data['is_holiday'] == 0]['sales_revenue'].mean()
            overall_effect = holiday_sales - non_holiday_sales

            # 不同节假日的效应
            holiday_effects = {}
            if 'holiday_name' in data.columns:
                holiday_groups = data[data['is_holiday'] == 1].groupby('holiday_name')['sales_revenue'].agg(['mean', 'count'])

                for holiday, stats in holiday_groups.iterrows():
                    if stats['count'] >= 3:  # 至少3个数据点
                        effect = stats['mean'] - non_holiday_sales
                        holiday_effects[holiday] = {
                            'effect': effect,
                            'sample_size': stats['count'],
                            'avg_sales': stats['mean']
                        }

            # 购物季分析
            shopping_season_effect = None
            if 'is_shopping_season' in data.columns:
                shopping_sales = data[data['is_shopping_season'] == 1]['sales_revenue'].mean()
                non_shopping_sales = data[data['is_shopping_season'] == 0]['sales_revenue'].mean()
                shopping_season_effect = {
                    'effect': shopping_sales - non_shopping_sales,
                    'interpretation': f"购物季销售额平均{'增加' if shopping_sales > non_shopping_sales else '减少'} ${abs(shopping_sales - non_shopping_sales):.0f}"
                }

            # 找出最佳和最差节假日
            if holiday_effects:
                best_holiday = max(holiday_effects.items(), key=lambda x: x[1]['effect'])
                worst_holiday = min(holiday_effects.items(), key=lambda x: x[1]['effect'])
            else:
                best_holiday = worst_holiday = None

            return {
                'overall_holiday_effect': overall_effect,
                'specific_holiday_effects': holiday_effects,
                'best_holiday': best_holiday[0] if best_holiday else None,
                'worst_holiday': worst_holiday[0] if worst_holiday else None,
                'shopping_season_effect': shopping_season_effect,
                'interpretation': f"节假日整体提升销售额 ${overall_effect:.0f}"
            }

        except Exception as e:
            logger.error(f"节假日效应分析失败: {e}")
            return {'error': str(e)}

    def _analyze_combined_effects(self) -> Dict[str, Any]:
        """分析组合效应"""
        logger.info("分析组合因果效应...")

        data = self.merged_data.copy()

        try:
            # 创建交互特征
            data['rainy_promotion'] = data['is_rainy'] * data['has_promotion']
            data['holiday_promotion'] = data['is_holiday'] * data['has_promotion']
            data['extreme_weather_holiday'] = data['is_extreme_weather'] * data['is_holiday']

            # 使用线性回归分析交互效应
            from sklearn.linear_model import LinearRegression

            features = [
                'has_promotion', 'is_rainy', 'is_holiday', 'is_extreme_weather',
                'rainy_promotion', 'holiday_promotion', 'extreme_weather_holiday',
                'day_of_week', 'temp_avg_f'
            ]

            # 确保所有特征都存在
            available_features = [f for f in features if f in data.columns]

            X = data[available_features]
            y = data['sales_revenue']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            coefficients = pd.DataFrame({
                'feature': available_features,
                'coefficient': model_lr.coef_
            }).sort_values('coefficient', key=abs, ascending=False)

            # 解释交互效应
            interactions = {}
            if 'rainy_promotion' in coefficients['feature'].values:
                interactions['rainy_promotion'] = {
                    'effect': coefficients[coefficients['feature'] == 'rainy_promotion']['coefficient'].values[0],
                    'interpretation': '雨天促销的额外效应'
                }

            if 'holiday_promotion' in coefficients['feature'].values:
                interactions['holiday_promotion'] = {
                    'effect': coefficients[coefficients['feature'] == 'holiday_promotion']['coefficient'].values[0],
                    'interpretation': '节假日促销的额外效应'
                }

            return {
                'feature_importance': coefficients.to_dict('records'),
                'interaction_effects': interactions,
                'model_r2': model_lr.score(X, y),
                'interpretation': self._interpret_combined_effects(coefficients)
            }

        except Exception as e:
            logger.error(f"组合效应分析失败: {e}")
            return {'error': str(e)}

    def perform_counterfactual_analysis(self) -> Dict[str, Any]:
        """执行反事实分析"""
        logger.info("开始反事实分析...")

        # 定义反事实场景
        scenarios = [
            {
                'name': '完美促销日：节假日+好天气+促销',
                'conditions': {
                    'is_holiday': 1,
                    'has_promotion': 1,
                    'is_rainy': 0,
                    'temp_avg_f': 72,
                    'is_extreme_weather': 0
                }
            },
            {
                'name': '最差情况：极端天气+无促销',
                'conditions': {
                    'is_extreme_weather': 1,
                    'has_promotion': 0,
                    'temp_avg_f': 100,
                    'is_rainy': 0
                }
            },
            {
                'name': '雨天促销策略',
                'conditions': {
                    'is_rainy': 1,
                    'has_promotion': 1,
                    'temp_avg_f': 60
                }
            },
            {
                'name': '黑色星期五场景',
                'conditions': {
                    'is_shopping_season': 1,
                    'has_promotion': 1,
                    'is_holiday': 1,
                    'day_of_week': 4  # 星期五
                }
            },
            {
                'name': '普通工作日',
                'conditions': {
                    'is_holiday': 0,
                    'has_promotion': 0,
                    'is_rainy': 0,
                    'day_of_week': 2,  # 星期三
                    'temp_avg_f': 70
                }
            }
        ]

        # 执行场景分析
        results = {}
        for scenario in scenarios:
            prediction = self._predict_counterfactual(scenario['conditions'])
            results[scenario['name']] = {
                'conditions': scenario['conditions'],
                'predicted_sales': prediction['sales'],
                'confidence_interval': prediction['ci'],
                'vs_average': prediction['sales'] - self.merged_data['sales_revenue'].mean()
            }

        # What-if 分析矩阵
        what_if_matrix = self._create_what_if_matrix()

        # 寻找最优条件
        optimal_conditions = self._find_optimal_conditions()

        # 识别风险场景
        risk_scenarios = self._identify_risk_scenarios()

        return {
            'scenario_analysis': results,
            'what_if_matrix': what_if_matrix,
            'optimal_conditions': optimal_conditions,
            'risk_scenarios': risk_scenarios
        }

    def _predict_counterfactual(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """预测反事实场景"""
        # 准备训练数据
        data = self.merged_data.copy()

        # 基础特征
        feature_cols = [
            'has_promotion', 'is_rainy', 'is_holiday', 'is_extreme_weather',
            'day_of_week', 'is_weekend'
        ]

        # 添加温度特征（如果存在）
        if 'temp_avg_f' in data.columns:
            feature_cols.append('temp_avg_f')

        # 过滤存在的特征
        available_features = [f for f in feature_cols if f in data.columns]

        X = data[available_features]
        y = data['sales_revenue']

        # 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)

        # 创建预测数据
        pred_dict = {
            'has_promotion': conditions.get('has_promotion', 0),
            'is_rainy': conditions.get('is_rainy', 0),
            'is_holiday': conditions.get('is_holiday', 0),
            'is_extreme_weather': conditions.get('is_extreme_weather', 0),
            'day_of_week': conditions.get('day_of_week', 3),
            'is_weekend': 1 if conditions.get('day_of_week', 3) >= 5 else 0
        }

        if 'temp_avg_f' in available_features:
            pred_dict['temp_avg_f'] = conditions.get('temp_avg_f', 70)

        # 只保留可用的特征
        pred_dict = {k: v for k, v in pred_dict.items() if k in available_features}
        pred_data = pd.DataFrame([pred_dict])

        # 预测
        prediction = rf_model.predict(pred_data)[0]

        # 获取预测区间
        all_predictions = np.array([tree.predict(pred_data)[0] for tree in rf_model.estimators_])
        ci_lower = np.percentile(all_predictions, 5)
        ci_upper = np.percentile(all_predictions, 95)

        return {
            'sales': prediction,
            'ci': (ci_lower, ci_upper)
        }

    def _create_what_if_matrix(self) -> pd.DataFrame:
        """创建What-if分析矩阵"""
        # 创建参数网格
        promotions = [0, 1]
        weather_conditions = ['晴天', '雨天', '极端天气']
        holidays = [0, 1]

        results = []

        for promo in promotions:
            for weather in weather_conditions:
                for holiday in holidays:
                    # 设置天气条件
                    if weather == '晴天':
                        conditions = {
                            'has_promotion': promo,
                            'is_rainy': 0,
                            'is_extreme_weather': 0,
                            'is_holiday': holiday,
                            'temp_avg_f': 72
                        }
                    elif weather == '雨天':
                        conditions = {
                            'has_promotion': promo,
                            'is_rainy': 1,
                            'is_extreme_weather': 0,
                            'is_holiday': holiday,
                            'temp_avg_f': 60
                        }
                    else:  # 极端天气
                        conditions = {
                            'has_promotion': promo,
                            'is_rainy': 0,
                            'is_extreme_weather': 1,
                            'is_holiday': holiday,
                            'temp_avg_f': 95
                        }

                    prediction = self._predict_counterfactual(conditions)

                    results.append({
                        '促销': '是' if promo else '否',
                        '天气': weather,
                        '节假日': '是' if holiday else '否',
                        '预测销售额': f"${prediction['sales']:.0f}",
                        '置信区间': f"${prediction['ci'][0]:.0f} - ${prediction['ci'][1]:.0f}"
                    })

        return pd.DataFrame(results)

    def _find_optimal_conditions(self) -> Dict[str, Any]:
        """寻找最优条件组合"""
        best_sales = 0
        best_conditions = None

        # 网格搜索
        for temp in range(60, 85, 5):  # 60-80°F
            for promo in [0, 1]:
                for holiday in [0, 1]:
                    for dow in range(7):  # 一周每天
                        conditions = {
                            'has_promotion': promo,
                            'is_rainy': 0,
                            'is_holiday': holiday,
                            'temp_avg_f': temp,
                            'is_extreme_weather': 0,
                            'day_of_week': dow
                        }

                        prediction = self._predict_counterfactual(conditions)

                        if prediction['sales'] > best_sales:
                            best_sales = prediction['sales']
                            best_conditions = conditions

        # 生成建议
        dow_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        recommendations = []

        if best_conditions['has_promotion']:
            recommendations.append("实施促销活动")
        if best_conditions['is_holiday']:
            recommendations.append("充分利用节假日流量")
        recommendations.append(f"最佳销售日是{dow_names[best_conditions['day_of_week']]}")
        recommendations.append(f"理想温度约{best_conditions['temp_avg_f']}°F")

        return {
            'conditions': best_conditions,
            'expected_sales': best_sales,
            'recommendations': recommendations
        }

    def _identify_risk_scenarios(self) -> List[Dict[str, Any]]:
        """识别风险场景"""
        avg_sales = self.merged_data['sales_revenue'].mean()

        risk_scenarios = [
            {
                'scenario': '极端高温(>100°F)+无促销',
                'conditions': {'temp_avg_f': 105, 'has_promotion': 0, 'is_extreme_weather': 1}
            },
            {
                'scenario': '暴风雪天气',
                'conditions': {'is_extreme_weather': 1, 'temp_avg_f': 25, 'is_rainy': 1}
            },
            {
                'scenario': '连续雨天+工作日',
                'conditions': {'is_rainy': 1, 'day_of_week': 2, 'has_promotion': 0}
            }
        ]

        results = []
        for scenario in risk_scenarios:
            prediction = self._predict_counterfactual(scenario['conditions'])
            loss = avg_sales - prediction['sales']

            # 建议缓解措施
            if '高温' in scenario['scenario']:
                mitigation = "加强空调、提供冷饮优惠、延长营业时间至晚上"
            elif '暴风雪' in scenario['scenario']:
                mitigation = "提前备货、加强外卖服务、员工安全保障"
            else:
                mitigation = "雨天专属优惠、加强线上推广、改善店内体验"

            results.append({
                'scenario': scenario['scenario'],
                'predicted_loss': loss,
                'risk_level': '高' if loss > avg_sales * 0.3 else '中',
                'mitigation': mitigation
            })

        return sorted(results, key=lambda x: x['predicted_loss'], reverse=True)

    def visualize_results(self, causal_results: Dict[str, Any], counterfactual_results: Dict[str, Any]):
        """可视化分析结果"""
        logger.info("生成可视化...")

        # 创建图表布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '天气对销售的影响', '促销效果分析',
                '节假日销售表现', '反事实场景对比',
                'What-if分析热力图', '风险场景分析'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )

        # 1. 天气影响（如果有数据）
        if 'weather_effect' in causal_results and 'temperature_effect' in causal_results['weather_effect']:
            temp_effect = causal_results['weather_effect']['temperature_effect']
            if 'temperature_impact' in temp_effect and 'mean' in temp_effect['temperature_impact']:
                temp_data = pd.DataFrame(temp_effect['temperature_impact']['mean'].items(),
                                        columns=['温度区间', '平均销售额'])
                fig.add_trace(
                    go.Bar(x=temp_data['温度区间'], y=temp_data['平均销售额'], name='温度影响'),
                    row=1, col=1
                )

        # 2. 促销效果趋势
        if self.merged_data is not None:
            promo_trend = self.merged_data.groupby(['date', 'has_promotion'])['sales_revenue'].mean().reset_index()
            for promo in [0, 1]:
                data = promo_trend[promo_trend['has_promotion'] == promo]
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=data['sales_revenue'],
                        name=f'{"有促销" if promo else "无促销"}',
                        mode='lines'
                    ),
                    row=1, col=2
                )

        # 3. 节假日效果
        if 'holiday_effect' in causal_results and 'specific_holiday_effects' in causal_results['holiday_effect']:
            holiday_effects = causal_results['holiday_effect']['specific_holiday_effects']
            if holiday_effects:
                holidays = list(holiday_effects.keys())[:10]  # 最多显示10个
                effects = [holiday_effects[h]['effect'] for h in holidays]

                fig.add_trace(
                    go.Bar(x=holidays, y=effects, name='节假日效果'),
                    row=2, col=1
                )

        # 4. 反事实场景对比
        if 'scenario_analysis' in counterfactual_results:
            scenarios = list(counterfactual_results['scenario_analysis'].keys())
            predictions = [v['predicted_sales'] for v in counterfactual_results['scenario_analysis'].values()]
            vs_average = [v['vs_average'] for v in counterfactual_results['scenario_analysis'].values()]

            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=predictions,
                    name='预测销售额',
                    text=[f"${p:.0f}" for p in predictions],
                    textposition='auto'
                ),
                row=2, col=2
            )

        # 5. What-if矩阵表格
        if 'what_if_matrix' in counterfactual_results:
            matrix = counterfactual_results['what_if_matrix'].head(8)  # 显示前8行
            fig.add_trace(
                go.Table(
                    header=dict(values=list(matrix.columns)),
                    cells=dict(values=[matrix[col] for col in matrix.columns])
                ),
                row=3, col=1
            )

        # 6. 风险场景
        if 'risk_scenarios' in counterfactual_results:
            risks = counterfactual_results['risk_scenarios']
            risk_names = [r['scenario'] for r in risks]
            risk_losses = [r['predicted_loss'] for r in risks]

            fig.add_trace(
                go.Bar(
                    x=risk_names,
                    y=risk_losses,
                    name='预计损失',
                    marker_color='red',
                    text=[f"${l:.0f}" for l in risk_losses],
                    textposition='auto'
                ),
                row=3, col=2
            )

        # 更新布局
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="FBR美国门店销售因果分析报告"
        )

        # 保存图表
        fig.write_html("fbr_us_causal_analysis_report.html")
        logger.info("可视化报告已保存至 fbr_us_causal_analysis_report.html")

        # 生成文字报告
        self._generate_text_report(causal_results, counterfactual_results)

    def _generate_text_report(self, causal_results: Dict[str, Any], counterfactual_results: Dict[str, Any]):
        """生成文字分析报告"""
        report = f"""
# FBR美国门店销售数据因果分析报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、执行摘要

本报告基于FBR美国门店的销售数据，结合天气和节假日信息，进行了深入的因果分析。

### 门店覆盖：
"""

        for store in self.store_locations:
            report += f"- {store['name']} (邮编: {store['postal_code']})\n"

        report += "\n### 关键发现：\n\n"

        # 1. 天气影响
        if 'weather_effect' in causal_results:
            weather = causal_results['weather_effect']
            if 'rainy_day_effect' in weather and 'effect' in weather['rainy_day_effect']:
                report += f"1. **天气影响**：\n"
                report += f"   - {weather['rainy_day_effect'].get('interpretation', '雨天对销售有显著影响')}\n"

            if 'temperature_effect' in weather and 'optimal_temperature' in weather['temperature_effect']:
                report += f"   - 最佳销售温度区间：{weather['temperature_effect']['optimal_temperature']}\n"

            if 'extreme_weather_effect' in weather and 'effect' in weather['extreme_weather_effect']:
                report += f"   - {weather['extreme_weather_effect'].get('interpretation', '极端天气影响销售')}\n"

        # 2. 促销效果
        if 'promotion_effect' in causal_results and 'average_effect' in causal_results['promotion_effect']:
            promo = causal_results['promotion_effect']
            report += f"\n2. **促销效果**：\n"
            report += f"   - {promo.get('interpretation', '促销活动对销售有正向影响')}\n"

            if 'heterogeneous_effects' in promo:
                report += f"   - 异质性效应：\n"
                for context, effect in promo['heterogeneous_effects'].items():
                    report += f"     * {context}: ${effect:.0f}\n"

        # 3. 节假日效应
        if 'holiday_effect' in causal_results:
            holiday = causal_results['holiday_effect']
            if 'overall_holiday_effect' in holiday:
                report += f"\n3. **节假日效应**：\n"
                report += f"   - {holiday.get('interpretation', '节假日影响销售')}\n"

                if holiday.get('best_holiday'):
                    report += f"   - 最佳销售节假日：{holiday['best_holiday']}\n"
                if holiday.get('worst_holiday'):
                    report += f"   - 销售最差节假日：{holiday['worst_holiday']}\n"

                if 'shopping_season_effect' in holiday and holiday['shopping_season_effect']:
                    report += f"   - 购物季效应：{holiday['shopping_season_effect']['interpretation']}\n"

        # 反事实分析
        report += "\n## 二、反事实分析\n\n### 场景分析：\n"

        if 'scenario_analysis' in counterfactual_results:
            for scenario_name, result in counterfactual_results['scenario_analysis'].items():
                report += f"\n**{scenario_name}**\n"
                report += f"- 预测销售额：${result['predicted_sales']:.0f}\n"
                report += f"- 相比平均值：${result['vs_average']:+.0f}\n"

        # 最优条件
        if 'optimal_conditions' in counterfactual_results:
            optimal = counterfactual_results['optimal_conditions']
            report += f"\n### 最优运营条件：\n"
            report += f"- 预期销售额：${optimal['expected_sales']:.0f}\n"
            report += "- 建议：\n"
            for rec in optimal['recommendations']:
                report += f"  * {rec}\n"

        # 风险预警
        if 'risk_scenarios' in counterfactual_results:
            report += "\n### 风险场景预警：\n"
            for risk in counterfactual_results['risk_scenarios'][:3]:
                report += f"\n- **{risk['scenario']}**\n"
                report += f"  * 预计损失：${risk['predicted_loss']:.0f}\n"
                report += f"  * 风险等级：{risk['risk_level']}\n"
                report += f"  * 缓解措施：{risk['mitigation']}\n"

        # 业务建议
        report += """

## 三、业务建议

### 基于因果分析的行动建议：

1. **天气应对策略**：
   - 建立天气监测预警系统，提前3-5天预测销售趋势
   - 雨天和极端天气时，加强外卖/配送服务
   - 在最佳温度区间（舒适天气）时，可以举办户外促销活动

2. **促销优化**：
   - 基于ROI分析，在周末和节假日加大促销力度
   - 雨天促销可以有效缓解天气带来的负面影响
   - 建议建立动态促销系统，根据天气和客流自动调整

3. **节假日运营**：
   - 重点准备主要节假日（感恩节、圣诞节等）的库存
   - 购物季（黑色星期五、网络星期一）需要特别准备
   - 针对表现较差的节假日，考虑特殊营销策略

4. **风险管理**：
   - 建立极端天气应急预案
   - 优化库存管理，避免因天气导致的损失
   - 考虑天气保险等风险对冲工具

## 四、数据说明

- 分析时间范围：最近365天
- 数据记录数：{len(self.merged_data) if self.merged_data is not None else 0}
- 使用方法：因果推断（DoWhy）、机器学习（随机森林）

## 五、后续行动

1. 实施A/B测试验证关键发现
2. 建立实时监控仪表板
3. 每月更新模型和分析
4. 扩展到更多门店和地区

---
报告结束
"""

        # 保存报告
        with open('fbr_us_causal_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info("文字报告已保存至 fbr_us_causal_analysis_report.txt")
        print("\n" + "="*50)
        print("分析完成！")
        print("="*50)
        print(report[:1500] + "...")  # 打印前1500字符

    # 辅助方法
    def _analyze_heterogeneous_effects(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, float]:
        """分析异质性效应"""
        heterogeneous = {}

        # 按周末/工作日分组
        for is_weekend in [0, 1]:
            subset = data[data['is_weekend'] == is_weekend]
            treated = subset[subset[treatment] == 1][outcome].mean()
            control = subset[subset[treatment] == 0][outcome].mean()
            effect = treated - control
            heterogeneous[f'{"周末" if is_weekend else "工作日"}'] = effect

        # 按天气分组
        if 'is_rainy' in data.columns:
            for is_rainy in [0, 1]:
                subset = data[data['is_rainy'] == is_rainy]
                if len(subset[subset[treatment] == 1]) > 0 and len(subset[subset[treatment] == 0]) > 0:
                    treated = subset[subset[treatment] == 1][outcome].mean()
                    control = subset[subset[treatment] == 0][outcome].mean()
                    effect = treated - control
                    heterogeneous[f'{"雨天" if is_rainy else "晴天"}'] = effect

        return heterogeneous

    def _interpret_combined_effects(self, coefficients: pd.DataFrame) -> str:
        """解释组合效应"""
        top_features = coefficients.head(3)

        interpretation = "综合分析显示：\n"
        for _, row in top_features.iterrows():
            feature = row['feature']
            coef = row['coefficient']

            # 翻译特征名
            feature_names = {
                'has_promotion': '促销活动',
                'is_rainy': '雨天',
                'is_holiday': '节假日',
                'is_extreme_weather': '极端天气',
                'temp_avg_f': '平均温度',
                'rainy_promotion': '雨天促销交互',
                'holiday_promotion': '节假日促销交互'
            }

            feature_cn = feature_names.get(feature, feature)
            effect = "正向" if coef > 0 else "负向"
            interpretation += f"- {feature_cn} 对销售额有{effect}影响（系数：{coef:.2f}）\n"

        return interpretation