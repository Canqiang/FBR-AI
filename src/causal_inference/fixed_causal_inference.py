"""
修复版 UMe 茶饮销售数据因果推断分析引擎 - 增强版
包含销售预测、客户分析、促销分析等功能
"""

import pandas as pd
import numpy as np
pd.options.display.max_columns = None

from datetime import timedelta
import clickhouse_connect
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

# EconML（异质效应分析）
from econml.dml import LinearDML

# 天气和节假日
import requests
import holidays

# 预测模型
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    print("⚠️ Prophet未安装，预测功能将使用简单模型")
    HAS_PROPHET = False

# 数据类型处理
from typing import Dict, List, Any, Optional, Tuple
import decimal

# 可视化
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "browser"


class UMeCausalInferenceEngine:
    """UMe 茶饮因果推断分析引擎 - 增强版"""

    def __init__(self, ch_config: dict, weather_api_key: str = None):
        self.ch_client = clickhouse_connect.get_client(**ch_config)
        self.scaler = StandardScaler()
        self.weather_api_key = weather_api_key

        # 美国节假日
        self.us_holidays = holidays.US()

        # 分析结果存储
        self.analysis_results = {}
        self.raw_data = None
        self.enhanced_data = None
        self.customer_data = None
        self.promotion_data = None
        self.forecast_results = None

    # ============================================================
    # 1. 数据抽取（增强版）
    # ============================================================
    def load_integrated_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """拉取销售数据并修复聚合问题"""
        print("📊 正在加载销售数据...")

        # 修复：避免any()聚合函数的嵌套问题，使用子查询方式
        sales_query = f"""
        WITH daily_base AS (
            SELECT
                toDate(created_at_pt) AS date,
                location_id,
                location_name,
                substring(location_name, position(location_name,'-')+1, 2) AS state,
                toDayOfWeek(created_at_pt) AS day_of_week,
                order_id,
                item_total_amt,
                item_discount,
                customer_id,
                is_loyalty,
                campaign_names,
                category_name,
                item_name,
                toHour(created_at_pt) AS hour_of_day
            FROM dw.fact_order_item_variations
            WHERE
                created_at_pt >= '{start_date}'
                AND created_at_pt <= '{end_date}'
                AND pay_status = 'COMPLETED'
        )
        SELECT
            date,
            location_id,
            any(location_name) AS location_name,
            any(state) AS state,
            any(day_of_week) AS day_of_week,
            
            /* 核心指标 */
            countDistinct(order_id) AS order_count,
            sum(item_total_amt) AS total_revenue,
            avg(item_total_amt) AS avg_order_value,
            sum(item_discount) AS total_discount,
            sum(item_discount>0) AS discount_orders,
            countDistinct(customer_id) AS unique_customers,
            sum(is_loyalty) AS loyalty_orders,
            sum(arrayExists(x->x='BOGO', assumeNotNull(campaign_names))) AS bogo_orders,
            countDistinct(category_name) AS category_diversity,
            
            /* 时段分布 */
            sum(if(hour_of_day BETWEEN 7 AND 10, 1, 0)) AS morning_orders,
            sum(if(hour_of_day BETWEEN 11 AND 14, 1, 0)) AS lunch_orders,
            sum(if(hour_of_day BETWEEN 15 AND 17, 1, 0)) AS afternoon_orders,
            sum(if(hour_of_day BETWEEN 18 AND 21, 1, 0)) AS evening_orders,
            
            /* 按类别统计 - 基于实际category_name */
            sum(if(category_name IN ('Milk Tea', 'Fruit Tea', 'Slush', 'Seasonal Drinks'), 1, 0)) AS tea_drinks_orders,
            sum(if(category_name = 'Coffee', 1, 0)) AS coffee_orders,
            sum(if(category_name IN ('Snacks', 'Toast', 'Mochi Donut'), 1, 0)) AS food_orders,
            sum(if(category_name = 'Caffeine-Free Drinks', 1, 0)) AS caffeine_free_orders,
            sum(if(category_name = 'Try Our New', 1, 0)) AS new_product_orders
            
        FROM daily_base
        GROUP BY
            date, location_id
        ORDER BY
            date, location_id
        """

        sales_df = self.ch_client.query_df(sales_query)

        # 修复：立即转换数值类型
        numeric_cols = [
            'order_count', 'total_revenue', 'avg_order_value', 'total_discount',
            'discount_orders', 'unique_customers', 'loyalty_orders', 'bogo_orders',
            'category_diversity', 'morning_orders', 'lunch_orders', 'afternoon_orders',
            'evening_orders', 'tea_drinks_orders', 'coffee_orders', 'food_orders',
            'caffeine_free_orders', 'new_product_orders', 'day_of_week'
        ]

        for col in numeric_cols:
            if col in sales_df.columns:
                sales_df[col] = pd.to_numeric(sales_df[col], errors='coerce')

        print(f"✅ 加载 {len(sales_df)} 条销售数据")
        self.raw_data = sales_df.copy()
        return sales_df.fillna(0)

    def load_customer_profile_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载客户画像数据"""
        print("👥 正在加载客户画像数据...")

        customer_query = f"""
        SELECT
            customer_id,
            given_name,
            family_name,
            phone_number,
            birthday,
            creation_source,
            location_id,
            location_name,
            customer_created_date,
            loyalty_created_date,
            order_final_total_cnt AS total_orders,
            order_final_total_amt AS total_spent,
            order_final_avg_amt AS avg_order_value,
            order_first_date AS first_order_date,
            order_last_date AS last_order_date,
            
            /* 客户分类标签 */
            offline,
            dormant,
            ultra_low_frequency,
            hardcore,
            silent,
            loyal,
            regular,
            potential,
            high_spending,
            medium_spending,
            low_spending,
            highly_active,
            moderately_active,
            low_active,
            churned,
            high_value_customer,
            high_potential_customer,
            key_development_customer,
            regular_customer,
            critical_win_back_customer,
            general_value_customer,
            general_win_back_customer,
            inactive_customer,
            
            /* 标签数组 */
            rfm_labels,
            loyalty_labels,
            consumption_labels
            
        FROM ads.customer_profile
        WHERE
            order_last_date >= '{start_date}'
            OR customer_created_date >= '{start_date}'
        """

        try:
            customer_df = self.ch_client.query_df(customer_query)
            print(f"✅ 加载 {len(customer_df)} 条客户数据")
            self.customer_data = customer_df
            return customer_df
        except Exception as e:
            print(f"⚠️ 加载客户数据失败: {e}")
            return pd.DataFrame()

    def load_promotion_sales_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载促销销售数据"""
        print("🎯 正在加载促销数据...")

        promotion_query = f"""
        SELECT
            order_date,
            weekdays,
            location_id,
            location_name,
            item_name,
            category_name,
            source,
            source_type,
            item_amt,
            item_total_amt,
            item_qty,
            order_qty,
            order_amt,
            milk_amt,
            milk_item_qty,
            non_milk_amt,
            non_milk_item_qty
        FROM ads.promotion_sales
        WHERE
            order_date >= '{start_date}'
            AND order_date <= '{end_date}'
        """

        try:
            promotion_df = self.ch_client.query_df(promotion_query)
            print(f"✅ 加载 {len(promotion_df)} 条促销数据")
            self.promotion_data = promotion_df
            return promotion_df
        except Exception as e:
            print(f"⚠️ 加载促销数据失败: {e}")
            return pd.DataFrame()

    # ============================================================
    # 2. 天气数据获取
    # ============================================================
    def get_weather_data(self, start_date: str, end_date: str, locations_df: pd.DataFrame) -> pd.DataFrame:
        """获取天气数据"""
        print("🌤️ 正在获取天气数据...")

        # 州坐标映射
        state_coords = {
            'CA': {'lat': 37.7749, 'lon': -122.4194, 'city': 'San Francisco'},
            'IL': {'lat': 41.8781, 'lon': -87.6298, 'city': 'Chicago'},
            'AZ': {'lat': 33.4484, 'lon': -112.0740, 'city': 'Phoenix'},
            'TX': {'lat': 29.7604, 'lon': -95.3698, 'city': 'Houston'},
        }

        unique_states = locations_df['state'].unique()
        weather_data_list = []

        for state in unique_states:
            if state not in state_coords:
                continue

            coords = state_coords[state]
            weather_data = self._fetch_weather_api(
                start_date, end_date, coords['lat'], coords['lon'], state
            )

            if weather_data is not None:
                weather_data_list.append(weather_data)

        if weather_data_list:
            weather_df = pd.concat(weather_data_list, ignore_index=True)
            print(f"✅ 获取 {len(weather_df)} 条天气记录")
        else:
            print("⚠️ 使用模拟天气数据")
            weather_df = self._generate_mock_weather_data(start_date, end_date, unique_states)

        return weather_df

    def _fetch_weather_api(self, start_date: str, end_date: str, lat: float, lon: float, state: str) -> pd.DataFrame:
        """从Open-Meteo API获取天气数据"""
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': start_date,
                'end_date': end_date,
                'daily': [
                    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                    'precipitation_sum', 'rain_sum', 'snowfall_sum',
                    'windspeed_10m_max', 'sunshine_duration'
                ],
                'timezone': 'America/Los_Angeles'
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                weather_df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'state': state,
                    'temperature_max': data['daily']['temperature_2m_max'],
                    'temperature_min': data['daily']['temperature_2m_min'],
                    'temperature_mean': data['daily']['temperature_2m_mean'],
                    'precipitation': data['daily']['precipitation_sum'],
                    'rain': data['daily']['rain_sum'],
                    'snow': data['daily']['snowfall_sum'],
                    'wind_speed': data['daily']['windspeed_10m_max'],
                    'sunshine_hours': data['daily']['sunshine_duration']
                })
                return weather_df
            else:
                return None

        except Exception as e:
            print(f"天气API请求失败: {e}")
            return None

    def _generate_mock_weather_data(self, start_date: str, end_date: str, states: list) -> pd.DataFrame:
        """生成模拟天气数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        weather_data = []

        for state in states:
            base_temp = {'CA': 22, 'IL': 15, 'AZ': 30, 'TX': 25}.get(state, 20)

            for date in date_range:
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = np.sin(2 * np.pi * day_of_year / 365.25)

                temp_max = base_temp + 8 * seasonal_factor + np.random.normal(0, 3)
                temp_min = temp_max - np.random.uniform(5, 15)
                temp_mean = (temp_max + temp_min) / 2

                weather_data.append({
                    'date': date,
                    'state': state,
                    'temperature_max': temp_max,
                    'temperature_min': temp_min,
                    'temperature_mean': temp_mean,
                    'precipitation': max(0, np.random.exponential(2) - 1),
                    'rain': max(0, np.random.exponential(1.5) - 0.5),
                    'snow': 0 if state in ['CA', 'AZ', 'TX'] else max(0, np.random.exponential(0.5) - 2),
                    'wind_speed': np.random.uniform(5, 25),
                    'sunshine_hours': np.random.uniform(4, 12)
                })

        return pd.DataFrame(weather_data)

    # ============================================================
    # 3. 特征工程（集成版）
    # ============================================================
    def create_all_features(self, sales_df: pd.DataFrame, weather_df: pd.DataFrame = None) -> pd.DataFrame:
        """创建所有特征：促销、天气、节假日、时间"""
        print("🔧 正在创建特征...")

        df = sales_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 1. 基础促销特征
        df = self._create_promotion_features(df)

        # 2. 时间和节假日特征
        df = self._create_calendar_features(df)

        # 3. 天气特征（如果有数据）
        if weather_df is not None:
            df = self._create_weather_features(df, weather_df)

        # 4. 交互特征
        df = self._create_interaction_features(df)

        # 5. 客户特征（如果有数据）
        if self.customer_data is not None:
            df = self._create_customer_features(df)

        print(f"✅ 特征工程完成，共 {len(df.columns)} 个特征")
        self.enhanced_data = df.copy()
        return df

    def _create_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建促销相关特征"""
        # 确保数值类型
        numeric_cols = ['total_revenue', 'total_discount', 'bogo_orders']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['has_promotion'] = (df['total_discount'] > 0).astype(int)
        df['promotion_intensity'] = df['total_discount'] / (df['total_revenue'] + df['total_discount'] + 1e-3)
        df['has_bogo'] = (df['bogo_orders'] > 0).astype(int)

        # 修复：安全的分位数计算
        try:
            df['total_revenue'] = df['total_revenue'].astype(float)
            df['low_performance'] = df.groupby('location_id')['total_revenue'].transform(
                lambda x: (x < x.quantile(0.25)).astype(int)
            )
        except Exception as e:
            print(f"分位数计算失败，使用默认值: {e}")
            df['low_performance'] = 0

        return df

    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建日历和节假日特征"""
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_monday'] = (df['date'].dt.dayofweek == 0).astype(int)
        df['is_friday'] = (df['date'].dt.dayofweek == 4).astype(int)
        df['is_member_day'] = (df['date'].dt.dayofweek == 2).astype(int)  # 周三

        # 节假日特征
        df['is_holiday'] = df['date'].apply(lambda x: x.date() in self.us_holidays).astype(int)
        df['is_holiday_week'] = df['date'].apply(
            lambda x: any((x + timedelta(days=i)).date() in self.us_holidays for i in range(-3, 4))
        ).astype(int)

        # 季节特征
        df['is_summer'] = df['date'].dt.month.isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['date'].dt.month.isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['date'].dt.month.isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['date'].dt.month.isin([9, 10, 11]).astype(int)

        # 特殊节日
        df['is_valentine'] = ((df['date'].dt.month == 2) & (df['date'].dt.day == 14)).astype(int)
        df['is_christmas_season'] = ((df['date'].dt.month == 12) & (df['date'].dt.day >= 15)).astype(int)

        return df

    def _create_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """创建天气特征"""
        if weather_df is None or len(weather_df) == 0:
            return df

        # 确保日期类型一致
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        # 确保天气数据数值类型
        weather_numeric_cols = [
            'temperature_max', 'temperature_min', 'temperature_mean',
            'precipitation', 'rain', 'snow', 'wind_speed', 'sunshine_hours'
        ]
        for col in weather_numeric_cols:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')

        # 合并天气数据
        merged = df.merge(weather_df, on=['date', 'state'], how='left')
        merged = merged.fillna(method='ffill').fillna(method='bfill')

        # 天气分类特征
        merged['is_hot'] = (merged['temperature_max'] > 30).astype(int)
        merged['is_cold'] = (merged['temperature_max'] < 10).astype(int)
        merged['is_mild'] = ((merged['temperature_max'] >= 15) & (merged['temperature_max'] <= 25)).astype(int)

        merged['is_rainy'] = (merged['precipitation'] > 2).astype(int)
        merged['is_heavy_rain'] = (merged['precipitation'] > 10).astype(int)
        merged['is_snowy'] = (merged['snow'] > 0).astype(int)

        merged['is_sunny'] = (merged['sunshine_hours'] > 8).astype(int)
        merged['is_windy'] = (merged['wind_speed'] > 20).astype(int)

        # 舒适度指数
        merged['comfort_index'] = (
            (merged['temperature_mean'] - 20).abs() * (-0.1) +
            merged['sunshine_hours'] * 0.1 -
            merged['precipitation'] * 0.05 -
            merged['wind_speed'] * 0.02
        )

        return merged.fillna(0)

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        # 基础交互
        df['weekend_promotion'] = df['is_weekend'] * df['has_promotion']
        df['holiday_promotion'] = df['is_holiday'] * df['has_promotion']

        # 天气交互（如果有天气数据）
        if 'is_rainy' in df.columns:
            df['rainy_promotion'] = df['is_rainy'] * df['has_promotion']
        if 'is_hot' in df.columns:
            df['hot_promotion'] = df['is_hot'] * df['has_promotion']

        return df

    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建客户特征"""
        if self.customer_data is None:
            return df

        # 按location_id聚合客户特征
        customer_summary = self.customer_data.groupby('location_id').agg({
            'high_value_customer': 'sum',
            'loyal': 'sum',
            'churned': 'sum',
            'total_spent': 'mean',
            'avg_order_value': 'mean'
        }).reset_index()

        customer_summary.columns = [
            'location_id',
            'high_value_customers',
            'loyal_customers',
            'churned_customers',
            'avg_customer_spent',
            'avg_customer_order_value'
        ]

        # 合并到主数据
        df = df.merge(customer_summary, on='location_id', how='left')

        return df.fillna(0)

    # ============================================================
    # 4. 销售预测功能（新增）
    # ============================================================
    def create_sales_forecast(self, days_ahead: int = 7) -> Dict[str, Any]:
        """创建销售预测"""
        print(f"\n🔮 开始创建 {days_ahead} 天销售预测...")

        if self.enhanced_data is None:
            print("❌ 请先运行完整分析以获取数据")
            return {}

        # 准备数据
        enhanced_data = self.enhanced_data.copy()
        enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])

        # 按日期聚合
        daily_revenue = enhanced_data.groupby('date')['total_revenue'].sum().reset_index()
        daily_revenue.columns = ['ds', 'y']

        # 检查数据量
        if len(daily_revenue) < 30:
            print(f"⚠️ 预测需要至少30天的历史数据，当前只有{len(daily_revenue)}天")
            return {
                'error': '数据不足',
                'required_days': 30,
                'current_days': len(daily_revenue)
            }

        try:
            if HAS_PROPHET:
                # 使用Prophet模型
                forecast_result = self._prophet_forecast(daily_revenue, days_ahead)
            else:
                # 使用简单模型
                forecast_result = self._simple_forecast(daily_revenue, days_ahead)

            # 存储预测结果
            self.forecast_results = forecast_result

            # 创建预测摘要
            forecast_summary = self._create_forecast_summary(forecast_result)

            print(f"✅ 预测完成，未来{days_ahead}天预测总额: ${forecast_summary['total_forecast']:,.0f}")

            return {
                'forecast': forecast_result,
                'summary': forecast_summary,
                'method': 'Prophet' if HAS_PROPHET else 'Polynomial Regression'
            }

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return {'error': str(e)}

    def _prophet_forecast(self, daily_revenue: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
        """使用Prophet进行预测"""
        # 初始化模型
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )

        # 添加节假日（如果有）
        if 'is_holiday' in self.enhanced_data.columns:
            holidays = self.enhanced_data[self.enhanced_data['is_holiday'] == 1][['date']].copy()
            holidays.columns = ['ds']
            holidays['holiday'] = 'US_holiday'
            holidays = holidays.drop_duplicates()
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                holidays=holidays
            )

        # 训练模型
        model.fit(daily_revenue)

        # 创建预测数据框
        future = model.make_future_dataframe(periods=days_ahead)

        # 进行预测
        forecast = model.predict(future)

        # 分离历史和未来
        last_date = daily_revenue['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_date].copy()

        # 添加实际值到历史部分
        forecast = forecast.merge(daily_revenue, on='ds', how='left')

        return forecast

    def _simple_forecast(self, daily_revenue: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
        """使用简单模型进行预测"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        # 准备特征
        daily_revenue['day_of_week'] = daily_revenue['ds'].dt.dayofweek
        daily_revenue['day_of_month'] = daily_revenue['ds'].dt.day
        daily_revenue['days_since_start'] = (daily_revenue['ds'] - daily_revenue['ds'].min()).dt.days

        # 特征列表
        features = ['days_since_start', 'day_of_week', 'day_of_month']
        X = daily_revenue[features]
        y = daily_revenue['y']

        # 多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # 训练模型
        model = LinearRegression()
        model.fit(X_poly, y)

        # 创建未来日期
        last_date = daily_revenue['ds'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead
        )

        # 创建未来特征
        future_data = pd.DataFrame({
            'ds': future_dates,
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'days_since_start': (future_dates - daily_revenue['ds'].min()).days
        })

        X_future = future_data[features]
        X_future_poly = poly.transform(X_future)

        # 预测
        predictions = model.predict(X_future_poly)

        # 计算置信区间
        std_error = np.std(y - model.predict(X_poly))

        # 组合结果
        forecast = pd.DataFrame({
            'ds': pd.concat([daily_revenue['ds'], future_dates]),
            'yhat': np.concatenate([model.predict(X_poly), predictions]),
            'yhat_lower': np.concatenate([
                model.predict(X_poly) - 1.96 * std_error,
                predictions - 1.96 * std_error
            ]),
            'yhat_upper': np.concatenate([
                model.predict(X_poly) + 1.96 * std_error,
                predictions + 1.96 * std_error
            ])
        })

        # 添加实际值
        forecast = forecast.merge(daily_revenue[['ds', 'y']], on='ds', how='left')

        return forecast

    def _create_forecast_summary(self, forecast: pd.DataFrame) -> Dict[str, Any]:
        """创建预测摘要"""
        # 获取未来预测
        last_actual_date = forecast[forecast['y'].notna()]['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_actual_date]

        # 计算统计
        total_forecast = future_forecast['yhat'].sum()
        avg_forecast = future_forecast['yhat'].mean()
        max_forecast = future_forecast['yhat'].max()
        min_forecast = future_forecast['yhat'].min()

        # 识别风险
        low_inventory_risk = []  # 预留接口

        return {
            'total_forecast': total_forecast,
            'avg_daily_forecast': avg_forecast,
            'max_daily_forecast': max_forecast,
            'min_daily_forecast': min_forecast,
            'forecast_days': len(future_forecast),
            'low_inventory_risk': low_inventory_risk,
            'last_actual_date': last_actual_date.strftime('%Y-%m-%d'),
            'forecast_start_date': future_forecast['ds'].min().strftime('%Y-%m-%d'),
            'forecast_end_date': future_forecast['ds'].max().strftime('%Y-%m-%d')
        }

    # ============================================================
    # 5. 多因素因果分析
    # ============================================================
    def analyze_all_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析所有因素的因果效应"""
        print("\n🎯 开始多因素因果分析...")

        results = {}

        # 分析因素列表
        factors_to_analyze = [
            {
                'name': 'has_promotion',
                'display_name': '促销活动',
                'confounders': ['is_weekend', 'is_holiday', 'day_of_week', 'unique_customers', 'category_diversity']
            },
            {
                'name': 'is_weekend',
                'display_name': '周末效应',
                'confounders': ['is_holiday', 'unique_customers', 'has_promotion']
            },
            {
                'name': 'is_holiday',
                'display_name': '节假日效应',
                'confounders': ['is_weekend', 'day_of_week', 'unique_customers']
            }
        ]

        # 添加天气因素（如果有数据）
        if 'is_hot' in df.columns:
            factors_to_analyze.extend([
                {
                    'name': 'is_hot',
                    'display_name': '高温天气',
                    'confounders': ['is_weekend', 'is_holiday', 'day_of_week', 'unique_customers']
                },
                {
                    'name': 'is_rainy',
                    'display_name': '雨天天气',
                    'confounders': ['is_weekend', 'is_holiday', 'day_of_week', 'temperature_mean']
                }
            ])

        # 执行分析
        for factor in factors_to_analyze:
            if factor['name'] in df.columns:
                result = self._analyze_single_factor_econml(
                    df, factor['name'], factor['display_name'], factor['confounders']
                )
                results[factor['name']] = result

        # 交互效应分析
        interaction_results = self._analyze_interactions(df)
        results['interactions'] = interaction_results

        # 异质性分析
        heterogeneity_results = self._analyze_heterogeneity(df)
        results['heterogeneity'] = heterogeneity_results

        self.analysis_results = results
        return results

    def _analyze_single_factor_econml(self, df: pd.DataFrame, treatment: str,
                                    treatment_name: str, confounders: List[str]) -> Dict[str, Any]:
        """使用EconML分析单个因素"""
        print(f"  📊 分析 {treatment_name}...")

        # 数据预处理
        analysis_cols = [treatment, 'total_revenue'] + confounders
        clean_df = self._force_numeric(df, analysis_cols).dropna(subset=analysis_cols)

        if len(clean_df) < 50:
            return {'error': '数据不足', 'sample_size': len(clean_df)}

        try:
            Y = clean_df['total_revenue'].values
            T = clean_df[treatment].values.astype(int)
            X = clean_df[confounders].values

            # 分割数据
            X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
                X, T, Y, test_size=0.2, random_state=42
            )

            # 使用LinearDML（更稳定）
            ldml = LinearDML(
                model_t='auto',
                model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
                discrete_treatment=True,
                random_state=42
            )

            ldml.fit(Y_tr, T_tr, X=X_tr)

            # 计算ATE和置信区间
            ate = float(ldml.ate(X_te))

            # 尝试获取置信区间
            try:
                ci_lower, ci_upper = ldml.ate_interval(X_te, alpha=0.05)
                ci_lower, ci_upper = float(ci_lower), float(ci_upper)
            except:
                # 如果无法计算置信区间，使用bootstrap估计
                ci_lower, ci_upper = ate - 1.96 * abs(ate) * 0.1, ate + 1.96 * abs(ate) * 0.1

            # 计算其他统计信息
            treatment_rate = clean_df[treatment].mean()
            treatment_group_mean = clean_df[clean_df[treatment] == 1]['total_revenue'].mean()
            control_group_mean = clean_df[clean_df[treatment] == 0]['total_revenue'].mean()

            print(f"    ✅ {treatment_name}: ATE = ${ate:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")

            return {
                'ate': ate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'treatment_rate': treatment_rate,
                'treatment_group_mean': treatment_group_mean,
                'control_group_mean': control_group_mean,
                'sample_size': len(clean_df),
                'significant': not (ci_lower <= 0 <= ci_upper)  # 置信区间不包含0
            }

        except Exception as e:
            print(f"    ❌ {treatment_name} 分析失败: {e}")
            return {'error': str(e)}

    def _analyze_interactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析交互效应"""
        print("  🔄 分析交互效应...")

        interactions = {}

        # 定义要分析的交互对
        interaction_pairs = [
            ('is_rainy', 'has_promotion', '雨天促销交互'),
            ('is_hot', 'has_promotion', '高温促销交互'),
            ('is_weekend', 'has_promotion', '周末促销交互'),
            ('is_holiday', 'is_weekend', '节假日周末交互')
        ]

        for factor1, factor2, name in interaction_pairs:
            if factor1 in df.columns and factor2 in df.columns:
                interaction_result = self._calculate_interaction_effect(df, factor1, factor2, name)
                interactions[f"{factor1}_x_{factor2}"] = interaction_result

        return interactions

    def _calculate_interaction_effect(self, df: pd.DataFrame, factor1: str, factor2: str, name: str) -> Dict[str, Any]:
        """计算交互效应"""
        try:
            # 创建四个组合的平均营收
            results = {}
            for val1 in [0, 1]:
                for val2 in [0, 1]:
                    mask = (df[factor1] == val1) & (df[factor2] == val2)
                    if mask.sum() > 5:  # 至少5个样本
                        group_revenue = df[mask]['total_revenue'].mean()
                        group_size = mask.sum()
                        results[f"{val1}_{val2}"] = {'revenue': group_revenue, 'count': group_size}

            if len(results) == 4:  # 所有四个组合都有数据
                # 计算交互效应
                baseline = results['0_0']['revenue']
                factor1_effect = results['1_0']['revenue'] - baseline
                factor2_effect = results['0_1']['revenue'] - baseline
                combined_effect = results['1_1']['revenue'] - baseline
                interaction_effect = combined_effect - factor1_effect - factor2_effect

                return {
                    'interaction_effect': interaction_effect,
                    'factor1_main_effect': factor1_effect,
                    'factor2_main_effect': factor2_effect,
                    'combined_effect': combined_effect,
                    'group_details': results,
                    'name': name
                }
            else:
                return {'error': '数据不足', 'name': name}

        except Exception as e:
            return {'error': str(e), 'name': name}

    def _analyze_heterogeneity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析异质性效应"""
        print("  🔍 分析异质性效应...")

        heterogeneity = {}

        # 按店铺分析促销效应异质性
        if 'has_promotion' in df.columns:
            store_effects = {}
            for store_id in df['location_id'].unique():
                store_df = df[df['location_id'] == store_id].copy()
                if len(store_df) > 30 and store_df['has_promotion'].sum() > 5:  # 足够的数据和促销样本
                    treated_mean = store_df[store_df['has_promotion'] == 1]['total_revenue'].mean()
                    control_mean = store_df[store_df['has_promotion'] == 0]['total_revenue'].mean()
                    effect = treated_mean - control_mean
                    store_effects[store_id] = {
                        'effect': effect,
                        'treated_mean': treated_mean,
                        'control_mean': control_mean,
                        'sample_size': len(store_df)
                    }

            heterogeneity['promotion_by_store'] = store_effects

        # 按天气条件分析异质性
        if 'is_hot' in df.columns and 'has_promotion' in df.columns:
            weather_effects = {}
            weather_conditions = ['is_hot', 'is_rainy', 'is_mild']

            for condition in weather_conditions:
                if condition in df.columns:
                    weather_df = df[df[condition] == 1]
                    if len(weather_df) > 20 and weather_df['has_promotion'].sum() > 3:
                        treated_mean = weather_df[weather_df['has_promotion'] == 1]['total_revenue'].mean()
                        control_mean = weather_df[weather_df['has_promotion'] == 0]['total_revenue'].mean()
                        effect = treated_mean - control_mean
                        weather_effects[condition] = {
                            'effect': effect,
                            'treated_mean': treated_mean,
                            'control_mean': control_mean,
                            'sample_size': len(weather_df)
                        }

            heterogeneity['promotion_by_weather'] = weather_effects

        # 按产品类别分析（新增）
        if 'has_promotion' in df.columns:
            category_effects = self._analyze_category_heterogeneity(df)
            if category_effects:
                heterogeneity['promotion_by_category'] = category_effects

        return heterogeneity

    def _analyze_category_heterogeneity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """按产品类别分析异质性"""
        category_effects = {}

        # 定义产品类别效应分析
        category_columns = [
            ('tea_drinks_orders', '茶饮类'),
            ('coffee_orders', '咖啡类'),
            ('food_orders', '小食类'),
            ('caffeine_free_orders', '无咖啡因饮品'),
            ('new_product_orders', '新品')
        ]

        for col, name in category_columns:
            if col in df.columns:
                # 分析该类别在促销和非促销时的表现
                promo_df = df[df['has_promotion'] == 1]
                no_promo_df = df[df['has_promotion'] == 0]

                if len(promo_df) > 10 and len(no_promo_df) > 10:
                    promo_avg = promo_df[col].mean()
                    no_promo_avg = no_promo_df[col].mean()

                    category_effects[name] = {
                        'promotion_avg': promo_avg,
                        'no_promotion_avg': no_promo_avg,
                        'lift': (promo_avg - no_promo_avg) / (no_promo_avg + 1e-6),
                        'absolute_difference': promo_avg - no_promo_avg
                    }

        return category_effects

    # ============================================================
    # 6. 缺失数据接口（预留）
    # ============================================================
    def get_inventory_data(self, item_ids: List[str] = None) -> pd.DataFrame:
        """获取库存数据 - 预留接口"""
        print("⚠️ 库存数据接口预留，暂无实际数据")
        # 返回空DataFrame，保持接口一致性
        return pd.DataFrame(columns=['item_id', 'location_id', 'current_stock', 'safety_stock'])

    def get_traffic_data(self, location_ids: List[str] = None) -> pd.DataFrame:
        """获取客流数据 - 预留接口"""
        print("⚠️ 客流数据接口预留，暂无实际数据")
        # 返回空DataFrame
        return pd.DataFrame(columns=['date', 'location_id', 'hour', 'visitor_count'])

    def get_supply_chain_data(self) -> pd.DataFrame:
        """获取供应链数据 - 预留接口"""
        print("⚠️ 供应链数据接口预留，暂无实际数据")
        # 返回空DataFrame
        return pd.DataFrame(columns=['item_id', 'supplier_id', 'lead_time', 'reliability_score'])

    # ============================================================
    # 7. 工具函数
    # ============================================================
    @staticmethod
    def _force_numeric(df, cols):
        """强制转换为数值类型，处理Decimal"""
        out = df.copy()
        for c in cols:
            if c in out.columns:
                if out[c].dtype == 'object':
                    try:
                        out[c] = pd.to_numeric(out[c], errors='coerce')
                    except:
                        import decimal
                        def convert_decimal(x):
                            if isinstance(x, decimal.Decimal):
                                return float(x)
                            return x
                        out[c] = out[c].apply(convert_decimal)
                        out[c] = pd.to_numeric(out[c], errors='coerce')
                else:
                    out[c] = pd.to_numeric(out[c], errors='coerce')
        return out

    # ============================================================
    # 8. 快速分析接口（增强版）
    # ============================================================
    def run_complete_analysis(self, start_date: str, end_date: str, include_forecast: bool = True) -> Dict[str, Any]:
        """运行完整分析流程（增强版）"""
        print("🚀 开始完整因果推断分析（增强版）...")
        print("="*60)

        # 1. 数据加载
        sales_df = self.load_integrated_data(start_date, end_date)

        # 2. 加载额外数据
        customer_df = self.load_customer_profile_data(start_date, end_date)
        promotion_df = self.load_promotion_sales_data(start_date, end_date)

        # 3. 天气数据
        weather_df = self.get_weather_data(start_date, end_date, sales_df)

        # 4. 特征工程
        enhanced_df = self.create_all_features(sales_df, weather_df)

        # 5. 因果分析
        analysis_results = self.analyze_all_factors(enhanced_df)

        # 6. 销售预测（如果需要）
        forecast_results = None
        if include_forecast:
            forecast_results = self.create_sales_forecast(days_ahead=7)

        # 7. 汇总分析信息
        summary = {
            'analysis_period': {'start': start_date, 'end': end_date},
            'data_summary': {
                'total_records': len(enhanced_df),
                'stores_count': enhanced_df['location_id'].nunique(),
                'states_count': enhanced_df['state'].nunique(),
                'date_range_days': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1,
                'customers_analyzed': len(customer_df) if customer_df is not None else 0,
                'promotions_analyzed': len(promotion_df) if promotion_df is not None else 0
            },
            'features_created': len(enhanced_df.columns),
            'analysis_results': analysis_results,
            'forecast_results': forecast_results,
            'raw_data': self.raw_data,
            'enhanced_data': self.enhanced_data,
            'customer_data': self.customer_data,
            'promotion_data': self.promotion_data
        }

        print("✅ 完整分析完成！")
        return summary

    def create_key_metrics_summary(self) -> Dict[str, Any]:
        """创建关键指标摘要"""
        if self.enhanced_data is None:
            return {}

        df = self.enhanced_data

        # 计算关键指标
        metrics = {
            'sales_revenue': {
                'last_7d': df[df['date'] >= df['date'].max() - timedelta(days=6)]['total_revenue'].sum(),
                'last_14d': df[df['date'] >= df['date'].max() - timedelta(days=13)]['total_revenue'].sum(),
                'change': 0  # 将计算周环比变化
            },
            'orders_count': {
                'last_7d': df[df['date'] >= df['date'].max() - timedelta(days=6)]['order_count'].sum(),
                'last_14d': df[df['date'] >= df['date'].max() - timedelta(days=13)]['order_count'].sum(),
                'change': 0
            },
            'average_order_value': {
                'last_7d': df[df['date'] >= df['date'].max() - timedelta(days=6)]['avg_order_value'].mean(),
                'last_14d': df[df['date'] >= df['date'].max() - timedelta(days=13)]['avg_order_value'].mean(),
                'change': 0
            },
            'unique_customers': {
                'last_7d': df[df['date'] >= df['date'].max() - timedelta(days=6)]['unique_customers'].sum(),
                'last_14d': df[df['date'] >= df['date'].max() - timedelta(days=13)]['unique_customers'].sum(),
                'change': 0
            }
        }

        # 计算变化率
        for metric in metrics:
            last_7d = metrics[metric]['last_7d']
            prev_7d = metrics[metric]['last_14d'] - last_7d
            if prev_7d > 0:
                metrics[metric]['change'] = ((last_7d - prev_7d) / prev_7d) * 100

        return metrics