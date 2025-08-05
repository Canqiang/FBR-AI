"""
修复版 UMe 茶饮因果推断分析 + HTML报告生成
修复问题：数据聚合、混淆变量、数据类型、报告输出
"""

import pandas as pd
import numpy as np

pd.options.display.max_columns = None

from datetime import datetime, timedelta
import clickhouse_connect
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings;

warnings.filterwarnings("ignore")

# 因果推断
from dowhy import CausalModel
import statsmodels.api as sm

# EconML（异质效应分析）
from econml.metalearners import TLearner
from econml.dml import CausalForestDML, LinearDML

# 天气数据API
import requests
import holidays
import decimal

# 可视化
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

# HTML报告生成
import base64
import json
from typing import Dict, List, Any, Optional


class FixedFBRCausalInference:
    """修复版 FBR 因果推断分析引擎"""

    def __init__(self, ch_config: dict, weather_api_key: str = None):
        self.ch_client = clickhouse_connect.get_client(**ch_config)
        self.scaler = StandardScaler()
        self.weather_api_key = weather_api_key
        self.us_holidays = holidays.US()

        # 存储分析结果用于报告生成
        self.analysis_results = {}
        self.data_summary = {}

    # ------------------------------------------------------------
    # 1. 修复的数据抽取逻辑
    # ------------------------------------------------------------
    def load_integrated_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """修复版：拉取并正确聚合数据"""
        print("📊 加载销售数据...")

        # 修复：直接聚合到日级别，避免小时级别的复杂性
        sales_query = f"""
        WITH daily_base AS (
            SELECT
                toDate(created_at_pt) AS date,
                location_id,
                location_name,
                substring(location_name, position(location_name,'-')+1, 2) AS state,
                toDayOfWeek(toDate(created_at_pt)) AS day_of_week,
                toMonth(toDate(created_at_pt)) AS month,

                -- 基础指标
                countDistinct(order_id) AS order_count,
                sum(item_total_amt) AS total_revenue,
                avg(item_total_amt) AS avg_order_value,
                sum(item_discount) AS total_discount,
                sum(if(item_discount > 0, 1, 0)) AS discount_orders,
                countDistinct(customer_id) AS unique_customers,
                sum(is_loyalty) AS loyalty_orders,
                sum(arrayExists(x->x='BOGO', assumeNotNull(campaign_names))) AS bogo_orders,
                countDistinct(category_name) AS category_diversity,

                -- 时段分析
                sum(if(toHour(created_at_pt) BETWEEN 7 AND 10, 1, 0)) AS morning_orders,
                sum(if(toHour(created_at_pt) BETWEEN 11 AND 14, 1, 0)) AS lunch_orders,
                sum(if(toHour(created_at_pt) BETWEEN 15 AND 17, 1, 0)) AS afternoon_orders,
                sum(if(toHour(created_at_pt) BETWEEN 18 AND 21, 1, 0)) AS evening_orders,

                -- 产品类型
                sum(if(category_name IN ('Milk Tea', 'Fruit Tea'), 1, 0)) AS cold_drink_orders,
                sum(if(category_name = 'Coffee', 1, 0)) AS hot_drink_orders,
                sum(if(category_name = 'Snacks', 1, 0)) AS food_orders,

                -- 产品营收
                sum(if(category_name IN ('Milk Tea', 'Fruit Tea'), item_total_amt, 0)) AS cold_drink_revenue,
                sum(if(category_name = 'Coffee', item_total_amt, 0)) AS hot_drink_revenue,
                sum(if(category_name = 'Snacks', item_total_amt, 0)) AS food_revenue

            FROM dw.fact_order_item_variations
            WHERE created_at_pt >= '{start_date}'
                AND created_at_pt <= '{end_date}'
                AND pay_status = 'COMPLETED'
            GROUP BY date, location_id, location_name, state, day_of_week, month
        )
        SELECT * FROM daily_base
        ORDER BY date, location_id
        """

        sales_df = self.ch_client.query_df(sales_query)

        # 立即转换数据类型，修复Decimal问题
        numeric_cols = [
            'order_count', 'total_revenue', 'avg_order_value', 'total_discount',
            'discount_orders', 'unique_customers', 'loyalty_orders', 'bogo_orders',
            'category_diversity', 'morning_orders', 'lunch_orders', 'afternoon_orders',
            'evening_orders', 'cold_drink_orders', 'hot_drink_orders', 'food_orders',
            'cold_drink_revenue', 'hot_drink_revenue', 'food_revenue',
            'day_of_week', 'month'
        ]

        for col in numeric_cols:
            if col in sales_df.columns:
                sales_df[col] = self._convert_decimal_to_float(sales_df[col])

        # 存储数据摘要
        self.data_summary = {
            'date_range': f"{start_date} ~ {end_date}",
            'total_records': len(sales_df),
            'unique_locations': sales_df['location_id'].nunique(),
            'total_revenue': float(sales_df['total_revenue'].sum()),
            'total_orders': int(sales_df['order_count'].sum()),
            'avg_daily_revenue': float(sales_df['total_revenue'].mean()),
            'promotion_rate': float((sales_df['total_discount'] > 0).mean())
        }

        print(f"✅ 加载完成: {len(sales_df)} 条日级记录")
        return sales_df.fillna(0)

    def _convert_decimal_to_float(self, series):
        """转换Decimal类型为float"""

        def convert_value(x):
            if isinstance(x, decimal.Decimal):
                return float(x)
            return x

        return pd.to_numeric(series.apply(convert_value), errors='coerce')

    # ------------------------------------------------------------
    # 2. 增强的特征工程
    # ------------------------------------------------------------
    def create_enhanced_treatment_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建增强版处理变量，包含更多混淆变量"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 基础促销变量
        df['has_promotion'] = (df['total_discount'] > 0).astype(int)
        df['promotion_intensity'] = df['total_discount'] / (df['total_revenue'] + df['total_discount'] + 1e-3)
        df['has_bogo'] = (df['bogo_orders'] > 0).astype(int)

        # 时间相关特征
        df['is_weekend'] = df['day_of_week'].isin([6, 7]).astype(int)
        df['is_member_day'] = (df['day_of_week'] == 3).astype(int)  # 周三会员日
        df['is_friday'] = (df['day_of_week'] == 5).astype(int)

        # 节假日特征
        df['is_holiday'] = df['date'].apply(lambda x: x.date() in self.us_holidays).astype(int)
        df['is_holiday_week'] = df['date'].apply(
            lambda x: any((x + timedelta(days=i)).date() in self.us_holidays for i in range(-3, 4))
        ).astype(int)

        # 季节性特征
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)

        # 业务指标相关特征
        df['high_customer_day'] = df.groupby('location_id')['unique_customers'].transform(
            lambda x: (x > x.quantile(0.75)).astype(int)
        )

        df['low_performance_day'] = df.groupby('location_id')['total_revenue'].transform(
            lambda x: (x < x.quantile(0.25)).astype(int)
        )

        # 产品组合特征
        df['cold_drink_ratio'] = df['cold_drink_orders'] / (df['order_count'] + 1e-3)
        df['hot_drink_ratio'] = df['hot_drink_orders'] / (df['order_count'] + 1e-3)
        df['food_ratio'] = df['food_orders'] / (df['order_count'] + 1e-3)

        # 忠诚度特征
        df['loyalty_rate'] = df['loyalty_orders'] / (df['order_count'] + 1e-3)
        df['high_loyalty_day'] = (df['loyalty_rate'] > df['loyalty_rate'].median()).astype(int)

        return df

    # ------------------------------------------------------------
    # 3. 改进的因果分析方法
    # ------------------------------------------------------------
    def analyze_promotion_effect_comprehensive(self, df: pd.DataFrame) -> dict:
        """改进版促销效应分析，使用更完整的混淆变量集"""
        print("\n🎯 开始综合促销效应分析...")

        results = {
            'method_comparison': {},
            'robustness_checks': {},
            'heterogeneity_analysis': {}
        }

        # 定义更完整的混淆变量集
        base_confounders = [
            'day_of_week', 'unique_customers', 'category_diversity',
            'loyalty_orders', 'is_weekend'
        ]

        enhanced_confounders = base_confounders + [
            'is_holiday', 'is_holiday_week', 'is_summer', 'is_winter',
            'high_customer_day', 'cold_drink_ratio', 'hot_drink_ratio',
            'loyalty_rate'
        ]

        # 1. DoWhy分析
        dowhy_results = self._analyze_promotion_dowhy_enhanced(df, enhanced_confounders)
        results['method_comparison']['DoWhy'] = dowhy_results

        # 2. EconML分析
        econml_results = self._analyze_promotion_econml_enhanced(df, enhanced_confounders)
        results['method_comparison']['EconML'] = econml_results

        # 3. 鲁棒性检验
        results['robustness_checks'] = self._robustness_checks(df, base_confounders, enhanced_confounders)

        # 4. 异质性分析
        results['heterogeneity_analysis'] = self._heterogeneity_analysis(df)

        # 存储结果
        self.analysis_results['promotion_effect'] = results

        return results

    def _analyze_promotion_dowhy_enhanced(self, df: pd.DataFrame, confounders: List[str]) -> dict:
        """增强版DoWhy分析"""
        print("  📊 DoWhy 因果推断分析...")

        treatment = 'has_promotion'
        outcome = 'total_revenue'

        # 数据预处理
        analysis_cols = [treatment, outcome] + confounders
        clean_df = self._force_numeric(df, analysis_cols).dropna(subset=analysis_cols)

        if len(clean_df) < 50:
            return {'error': '数据不足'}

        # 因果图（简化但更准确）
        graph = """
        digraph {
            is_member_day -> has_promotion;
            has_promotion -> total_revenue;

            day_of_week -> {has_promotion total_revenue};
            is_weekend -> {has_promotion total_revenue};
            is_holiday -> {has_promotion total_revenue};
            unique_customers -> total_revenue;
            loyalty_orders -> {has_promotion total_revenue};
            category_diversity -> total_revenue;
        }
        """

        try:
            model = CausalModel(clean_df, treatment, outcome, graph)
            ident = model.identify_effect(proceed_when_unidentifiable=True)

            results = {}

            # 线性回归（最稳定）
            lr = model.estimate_effect(ident, method_name="backdoor.linear_regression")
            results['LinearRegression'] = {
                'ate': float(lr.value),
                'confidence_interval': [float(lr.value - 1.96 * lr.stderr),
                                        float(lr.value + 1.96 * lr.stderr)] if lr.stderr else None
            }

            # PSM（如果样本足够）
            try:
                if len(clean_df) > 100:
                    psm = model.estimate_effect(ident, method_name="backdoor.propensity_score_matching")
                    results['PSM'] = {
                        'ate': float(psm.value),
                        'confidence_interval': [float(psm.value - 1.96 * psm.stderr),
                                                float(psm.value + 1.96 * psm.stderr)] if psm.stderr else None
                    }
            except Exception as e:
                print(f"    ⚠️ PSM分析失败: {e}")

            # 计算基础统计
            treatment_group = clean_df[clean_df[treatment] == 1][outcome].mean()
            control_group = clean_df[clean_df[treatment] == 0][outcome].mean()
            naive_diff = treatment_group - control_group

            results['descriptive_stats'] = {
                'treatment_group_mean': float(treatment_group),
                'control_group_mean': float(control_group),
                'naive_difference': float(naive_diff),
                'treatment_rate': float(clean_df[treatment].mean()),
                'sample_size': len(clean_df)
            }

            print(f"    ✅ DoWhy ATE: ${results['LinearRegression']['ate']:.2f}")

            return results

        except Exception as e:
            print(f"    ❌ DoWhy分析失败: {e}")
            return {'error': str(e)}

    def _analyze_promotion_econml_enhanced(self, df: pd.DataFrame, confounders: List[str]) -> dict:
        """增强版EconML分析"""
        print("  🤖 EconML 机器学习因果推断...")

        treatment = 'has_promotion'
        outcome = 'total_revenue'

        # 数据预处理
        analysis_cols = [treatment, outcome] + confounders
        clean_df = self._force_numeric(df, analysis_cols).dropna(subset=analysis_cols)

        if len(clean_df) < 50:
            return {'error': '数据不足'}

        try:
            Y = clean_df[outcome].values.astype(float)
            T = clean_df[treatment].values.astype(int)
            X = clean_df[confounders].values.astype(float)

            # 数据分割
            X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
                X, T, Y, test_size=0.2, random_state=42
            )

            # 使用LinearDML（更稳定）
            ldml = LinearDML(
                model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
                model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
                discrete_treatment=True,
                random_state=42
            )

            ldml.fit(Y_tr, T_tr, X=X_tr)
            ate = float(ldml.ate(X_te))

            # 计算置信区间
            ate_interval = ldml.ate_interval(X_te, alpha=0.05)

            # 个体效应
            cate = ldml.effect(X_te)

            results = {
                'LinearDML_ATE': ate,
                'confidence_interval': [float(ate_interval[0]), float(ate_interval[1])],
                'cate_std': float(np.std(cate)),
                'cate_range': [float(np.min(cate)), float(np.max(cate))],
                'positive_effect_rate': float((cate > 0).mean()),
                'sample_size': len(clean_df)
            }

            print(f"    ✅ EconML ATE: ${ate:.2f} [{ate_interval[0]:.2f}, {ate_interval[1]:.2f}]")

            return results

        except Exception as e:
            print(f"    ❌ EconML分析失败: {e}")
            return {'error': str(e)}

    def _robustness_checks(self, df: pd.DataFrame, base_confounders: List[str],
                           enhanced_confounders: List[str]) -> dict:
        """鲁棒性检验"""
        print("  🔍 鲁棒性检验...")

        results = {}

        # 1. 不同混淆变量集的对比
        base_result = self._simple_ate_estimation(df, base_confounders)
        enhanced_result = self._simple_ate_estimation(df, enhanced_confounders)

        results['confounder_sensitivity'] = {
            'base_confounders_ate': base_result,
            'enhanced_confounders_ate': enhanced_result,
            'difference': enhanced_result - base_result if base_result and enhanced_result else None
        }

        # 2. 不同时间段的稳定性
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        mid_point = len(df_sorted) // 2

        first_half = df_sorted.iloc[:mid_point]
        second_half = df_sorted.iloc[mid_point:]

        first_ate = self._simple_ate_estimation(first_half, base_confounders)
        second_ate = self._simple_ate_estimation(second_half, base_confounders)

        results['temporal_stability'] = {
            'first_period_ate': first_ate,
            'second_period_ate': second_ate,
            'difference': second_ate - first_ate if first_ate and second_ate else None
        }

        return results

    def _heterogeneity_analysis(self, df: pd.DataFrame) -> dict:
        """异质性分析"""
        print("  📈 异质性效应分析...")

        results = {}

        # 按州分析
        state_effects = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            if len(state_df) > 30:
                ate = self._simple_ate_estimation(state_df, ['day_of_week', 'unique_customers'])
                if ate:
                    state_effects[state] = ate

        results['by_state'] = state_effects

        # 按季节分析
        seasonal_effects = {}
        for season in ['summer', 'winter', 'spring', 'fall']:
            season_df = df[df[f'is_{season}'] == 1]
            if len(season_df) > 30:
                ate = self._simple_ate_estimation(season_df, ['day_of_week', 'unique_customers'])
                if ate:
                    seasonal_effects[season] = ate

        results['by_season'] = seasonal_effects

        # 按店铺规模分析（基于平均营收）
        df['store_size'] = pd.qcut(df.groupby('location_id')['total_revenue'].transform('mean'),
                                   q=3, labels=['Small', 'Medium', 'Large'])

        size_effects = {}
        for size in ['Small', 'Medium', 'Large']:
            size_df = df[df['store_size'] == size]
            if len(size_df) > 30:
                ate = self._simple_ate_estimation(size_df, ['day_of_week', 'unique_customers'])
                if ate:
                    size_effects[size] = ate

        results['by_store_size'] = size_effects

        return results

    def _simple_ate_estimation(self, df: pd.DataFrame, confounders: List[str]) -> Optional[float]:
        """简单的ATE估计（用于鲁棒性检验）"""
        try:
            from sklearn.linear_model import LinearRegression

            treatment = 'has_promotion'
            outcome = 'total_revenue'

            # 数据准备
            analysis_cols = [treatment, outcome] + confounders
            clean_df = self._force_numeric(df, analysis_cols).dropna(subset=analysis_cols)

            if len(clean_df) < 20:
                return None

            # 简单线性回归
            X = clean_df[[treatment] + confounders]
            y = clean_df[outcome]

            model = LinearRegression()
            model.fit(X, y)

            # 促销系数就是ATE估计
            ate = model.coef_[0]  # 第一个系数是处理变量系数

            return float(ate)

        except Exception:
            return None

    # ------------------------------------------------------------
    # 4. HTML报告生成
    # ------------------------------------------------------------
    def generate_html_report(self, output_file: str = None) -> str:
        """生成HTML分析报告"""
        print("\n📋 生成HTML分析报告...")

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"UMe_causal_analysis_report_{timestamp}.html"

        # 生成图表
        charts_html = self._generate_charts_html()

        # HTML模板
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMe 茶饮因果分析报告</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧋 UMe 茶饮因果分析报告</h1>
            <p class="subtitle">促销活动效应评估与业务洞察</p>
            <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        {self._generate_executive_summary_html()}

        {self._generate_data_overview_html()}

        {self._generate_causal_analysis_html()}

        {charts_html}

        {self._generate_robustness_checks_html()}

        {self._generate_business_insights_html()}

        <footer class="footer">
            <p>© 2025 UMe 茶饮数据分析团队 | 基于因果推断的科学决策支持</p>
        </footer>
    </div>
</body>
</html>
"""

        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML报告已生成: {output_file}")
        return output_file

    def _get_css_styles(self) -> str:
        """CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 10px;
        }

        .timestamp {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .section h3 {
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 15px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .results-table th {
            background-color: #3498db;
            color: white;
        }

        .results-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .insight-box {
            background: #e8f6f3;
            border-left: 4px solid #27ae60;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }

        .warning-box {
            background: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }

        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 40px;
        }

        .highlight {
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .positive {
            color: #27ae60;
            font-weight: bold;
        }

        .negative {
            color: #e74c3c;
            font-weight: bold;
        }

        ul {
            padding-left: 20px;
        }

        li {
            margin-bottom: 8px;
        }
        """

    def _generate_executive_summary_html(self) -> str:
        """生成执行摘要"""
        if 'promotion_effect' not in self.analysis_results:
            return ""

        results = self.analysis_results['promotion_effect']
        dowhy_ate = results.get('method_comparison', {}).get('DoWhy', {}).get('LinearRegression', {}).get('ate', 0)
        econml_ate = results.get('method_comparison', {}).get('EconML', {}).get('LinearDML_ATE', 0)

        avg_ate = (dowhy_ate + econml_ate) / 2 if dowhy_ate and econml_ate else (dowhy_ate or econml_ate or 0)

        promo_rate = self.data_summary.get('promotion_rate', 0)
        avg_revenue = self.data_summary.get('avg_daily_revenue', 0)

        impact_pct = (avg_ate / avg_revenue * 100) if avg_revenue > 0 else 0

        return f"""
        <div class="section">
            <h2>📊 执行摘要</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value {'positive' if avg_ate > 0 else 'negative'}">${avg_ate:.0f}</div>
                    <div class="stat-label">促销平均因果效应</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{promo_rate:.1%}</div>
                    <div class="stat-label">促销活动覆盖率</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value {'positive' if impact_pct > 0 else 'negative'}">{impact_pct:+.1f}%</div>
                    <div class="stat-label">对日均营收的影响</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.data_summary.get('total_records', 0)}</div>
                    <div class="stat-label">分析样本量</div>
                </div>
            </div>

            <div class="insight-box">
                <h3>🎯 关键发现</h3>
                <ul>
                    <li>促销活动对日均营收的<strong>因果效应为 ${avg_ate:.0f}</strong>，相当于 {impact_pct:+.1f}% 的影响</li>
                    <li>当前促销覆盖率为 {promo_rate:.1%}，{'建议适度扩大促销范围' if avg_ate > 0 and promo_rate < 0.3 else '当前促销频率较为合理'}</li>
                    <li>分析基于 {self.data_summary.get('total_records', 0)} 个店铺日级别观测，使用多种因果推断方法验证</li>
                    <li>{'促销活动显著提升营收，建议继续执行' if avg_ate > 50 else '促销效果有限，建议优化策略' if avg_ate > 0 else '促销可能存在负面影响，建议重新评估'}</li>
                </ul>
            </div>
        </div>
        """

    def _generate_data_overview_html(self) -> str:
        """生成数据概览"""
        return f"""
        <div class="section">
            <h2>📈 数据概览</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{self.data_summary.get('date_range', 'N/A')}</div>
                    <div class="stat-label">分析时间范围</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.data_summary.get('unique_locations', 0)}</div>
                    <div class="stat-label">门店数量</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${self.data_summary.get('total_revenue', 0):,.0f}</div>
                    <div class="stat-label">总营收</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.data_summary.get('total_orders', 0):,}</div>
                    <div class="stat-label">总订单数</div>
                </div>
            </div>
        </div>
        """

    def _generate_causal_analysis_html(self) -> str:
        """生成因果分析结果"""
        if 'promotion_effect' not in self.analysis_results:
            return "<div class='section'><h2>❌ 分析结果不可用</h2></div>"

        results = self.analysis_results['promotion_effect']
        method_results = results.get('method_comparison', {})

        # DoWhy结果
        dowhy_html = ""
        if 'DoWhy' in method_results and 'error' not in method_results['DoWhy']:
            dowhy = method_results['DoWhy']
            lr_result = dowhy.get('LinearRegression', {})
            ate = lr_result.get('ate', 0)
            ci = lr_result.get('confidence_interval', [])

            dowhy_html = f"""
            <h3>📊 DoWhy 因果推断结果</h3>
            <table class="results-table">
                <tr><th>方法</th><th>ATE估计</th><th>置信区间</th></tr>
                <tr>
                    <td>线性回归</td>
                    <td class="{'positive' if ate > 0 else 'negative'}">${ate:.2f}</td>
                    <td>{f'[${ci[0]:.2f}, ${ci[1]:.2f}]' if ci else 'N/A'}</td>
                </tr>
            </table>
            """

        # EconML结果
        econml_html = ""
        if 'EconML' in method_results and 'error' not in method_results['EconML']:
            econml = method_results['EconML']
            ate = econml.get('LinearDML_ATE', 0)
            ci = econml.get('confidence_interval', [])

            econml_html = f"""
            <h3>🤖 EconML 机器学习因果推断结果</h3>
            <table class="results-table">
                <tr><th>方法</th><th>ATE估计</th><th>置信区间</th><th>正向效应比例</th></tr>
                <tr>
                    <td>LinearDML</td>
                    <td class="{'positive' if ate > 0 else 'negative'}">${ate:.2f}</td>
                    <td>{f'[${ci[0]:.2f}, ${ci[1]:.2f}]' if ci else 'N/A'}</td>
                    <td>{econml.get('positive_effect_rate', 0):.1%}</td>
                </tr>
            </table>
            """

        return f"""
        <div class="section">
            <h2>🎯 因果分析结果</h2>
            {dowhy_html}
            {econml_html}

            <div class="insight-box">
                <h3>💡 方法对比解释</h3>
                <ul>
                    <li><strong>DoWhy</strong>: 基于因果图的推断，适合理论驱动的分析</li>
                    <li><strong>EconML</strong>: 基于机器学习的方法，能够捕捉复杂的非线性关系</li>
                    <li><strong>ATE (Average Treatment Effect)</strong>: 平均处理效应，表示促销对营收的平均因果影响</li>
                    <li><strong>置信区间</strong>: 表示估计的不确定性范围，95%置信水平</li>
                </ul>
            </div>
        </div>
        """

    def _generate_robustness_checks_html(self) -> str:
        """生成鲁棒性检验结果"""
        if 'promotion_effect' not in self.analysis_results:
            return ""

        robustness = self.analysis_results['promotion_effect'].get('robustness_checks', {})
        heterogeneity = self.analysis_results['promotion_effect'].get('heterogeneity_analysis', {})

        # 鲁棒性检验
        robustness_html = ""
        if robustness:
            conf_sens = robustness.get('confounder_sensitivity', {})
            temp_stab = robustness.get('temporal_stability', {})

            robustness_html = f"""
            <h3>🔍 鲁棒性检验</h3>
            <table class="results-table">
                <tr><th>检验项目</th><th>结果1</th><th>结果2</th><th>差异</th></tr>
                <tr>
                    <td>混淆变量敏感性</td>
                    <td>${conf_sens.get('base_confounders_ate', 0):.2f}</td>
                    <td>${conf_sens.get('enhanced_confounders_ate', 0):.2f}</td>
                    <td>${conf_sens.get('difference', 0):.2f}</td>
                </tr>
                <tr>
                    <td>时间稳定性</td>
                    <td>${temp_stab.get('first_period_ate', 0):.2f}</td>
                    <td>${temp_stab.get('second_period_ate', 0):.2f}</td>
                    <td>${temp_stab.get('difference', 0):.2f}</td>
                </tr>
            </table>
            """

        # 异质性分析
        heterogeneity_html = ""
        if heterogeneity:
            state_effects = heterogeneity.get('by_state', {})
            seasonal_effects = heterogeneity.get('by_season', {})

            if state_effects:
                state_rows = "".join([
                    f"<tr><td>{state}</td><td class=\"{'positive' if effect > 0 else 'negative'}\">${effect:.2f}</td></tr>"
                    for state, effect in state_effects.items()
                ])

                heterogeneity_html += f"""
                <h3>📍 各州促销效应差异</h3>
                <table class="results-table">
                    <tr><th>州</th><th>ATE估计</th></tr>
                    {state_rows}
                </table>
                """

        return f"""
        <div class="section">
            <h2>🛡️ 鲁棒性与异质性分析</h2>
            {robustness_html}
            {heterogeneity_html}

            <div class="warning-box">
                <h3>⚠️ 解读说明</h3>
                <ul>
                    <li><strong>混淆变量敏感性</strong>: 使用不同混淆变量集的结果差异，差异小说明结果稳定</li>
                    <li><strong>时间稳定性</strong>: 不同时间段的效应差异，评估结果的时间一致性</li>
                    <li><strong>异质性分析</strong>: 不同群体的效应差异，有助于精准营销策略</li>
                </ul>
            </div>
        </div>
        """

    def _generate_business_insights_html(self) -> str:
        """生成业务洞察"""
        if 'promotion_effect' not in self.analysis_results:
            return ""

        results = self.analysis_results['promotion_effect']
        dowhy_ate = results.get('method_comparison', {}).get('DoWhy', {}).get('LinearRegression', {}).get('ate', 0)
        econml_ate = results.get('method_comparison', {}).get('EconML', {}).get('LinearDML_ATE', 0)
        avg_ate = (dowhy_ate + econml_ate) / 2 if dowhy_ate and econml_ate else (dowhy_ate or econml_ate or 0)

        promo_rate = self.data_summary.get('promotion_rate', 0)
        avg_revenue = self.data_summary.get('avg_daily_revenue', 0)

        # 生成建议
        recommendations = []

        if avg_ate > 50:
            recommendations.append("✅ 促销活动效果显著，建议继续执行并适度扩大范围")
            if promo_rate < 0.3:
                recommendations.append("📈 当前促销覆盖率较低，有进一步提升空间")
        elif avg_ate > 0:
            recommendations.append("⚠️ 促销活动有正向效果但不够显著，建议优化促销策略")
            recommendations.append("🎯 考虑提高促销力度或改进促销方式")
        else:
            recommendations.append("🚨 促销活动可能存在负面影响，建议暂停并重新评估")
            recommendations.append("🔍 深入分析促销成本和品牌影响")

        # 异质性建议
        heterogeneity = results.get('heterogeneity_analysis', {})
        state_effects = heterogeneity.get('by_state', {})
        if state_effects:
            best_state = max(state_effects, key=state_effects.get)
            worst_state = min(state_effects, key=state_effects.get)
            recommendations.append(f"🌟 {best_state}州促销效果最佳，可作为标杆推广经验")
            if state_effects[worst_state] < 0:
                recommendations.append(f"⚠️ {worst_state}州促销效果较差，需要针对性改进")

        recommendations_html = "".join([f"<li>{rec}</li>" for rec in recommendations])

        return f"""
        <div class="section">
            <h2>💡 业务洞察与建议</h2>

            <div class="insight-box">
                <h3>🎯 核心洞察</h3>
                <ul>
                    <li>促销活动的<strong>真实因果效应</strong>为每天 ${avg_ate:.0f}，考虑了混淆因素后的净影响</li>
                    <li>这相当于对平均日营收产生 {(avg_ate / avg_revenue * 100) if avg_revenue > 0 else 0:.1f}% 的影响</li>
                    <li>当前 {promo_rate:.1%} 的促销覆盖率下，系统性促销策略的效果已被量化</li>
                </ul>
            </div>

            <div class="insight-box">
                <h3>🚀 行动建议</h3>
                <ul>
                    {recommendations_html}
                </ul>
            </div>

            <div class="warning-box">
                <h3>⚠️ 重要提醒</h3>
                <ul>
                    <li>本分析基于历史数据，实际效果可能因市场环境变化而异</li>
                    <li>建议结合A/B测试验证因果推断结果</li>
                    <li>持续监控促销效果，定期更新分析模型</li>
                    <li>考虑促销的长期影响，如品牌价值和客户习惯培养</li>
                </ul>
            </div>
        </div>
        """

    def _generate_charts_html(self) -> str:
        """生成图表HTML"""
        if 'promotion_effect' not in self.analysis_results:
            return ""

        # 创建效应对比图
        results = self.analysis_results['promotion_effect']
        method_results = results.get('method_comparison', {})

        methods = []
        values = []

        if 'DoWhy' in method_results and 'error' not in method_results['DoWhy']:
            ate = method_results['DoWhy'].get('LinearRegression', {}).get('ate', 0)
            methods.append('DoWhy-线性回归')
            values.append(ate)

        if 'EconML' in method_results and 'error' not in method_results['EconML']:
            ate = method_results['EconML'].get('LinearDML_ATE', 0)
            methods.append('EconML-LinearDML')
            values.append(ate)

        if not methods:
            return ""

        # 生成Plotly图表
        fig_data = {
            'data': [{
                'x': methods,
                'y': values,
                'type': 'bar',
                'text': [f'${v:.0f}' for v in values],
                'textposition': 'auto',
                'marker': {
                    'color': ['#3498db' if v > 0 else '#e74c3c' for v in values]
                }
            }],
            'layout': {
                'title': '促销因果效应估计对比',
                'xaxis': {'title': '分析方法'},
                'yaxis': {'title': '平均因果效应 ($)'},
                'showlegend': False
            }
        }

        chart_json = json.dumps(fig_data)

        return f"""
        <div class="section">
            <h2>📊 可视化分析</h2>
            <div class="chart-container">
                <div id="effectsChart" style="width:100%;height:400px;"></div>
            </div>

            <script>
                Plotly.newPlot('effectsChart', {chart_json});
            </script>
        </div>
        """

    # ------------------------------------------------------------
    # 5. 工具函数
    # ------------------------------------------------------------
    @staticmethod
    def _force_numeric(df, cols):
        """强制转换为数值类型，处理Decimal"""
        out = df.copy()
        for c in cols:
            if c in out.columns:
                # 处理Decimal类型
                def convert_decimal(x):
                    if isinstance(x, decimal.Decimal):
                        return float(x)
                    return x

                out[c] = out[c].apply(convert_decimal)
                out[c] = pd.to_numeric(out[c], errors='coerce')
        return out


# ──────────────────────────────────────────────
# 使用示例
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 配置
    CLICKHOUSE_CONFIG = dict(
        host="clickhouse-0-0.umetea.net",
        port=443,
        database="dw",
        user="ml_ume",
        password="hDAoDvg8x552bH",
        verify=False,
    )

    print("🚀 启动修复版 UMe 茶饮因果推断分析")
    print("=" * 60)

    # 初始化分析引擎
    analyzer = FixedFBRCausalInference(CLICKHOUSE_CONFIG)

    # 设置分析参数
    start_date, end_date = "2025-06-01", "2025-07-31"

    try:
        # 1. 数据加载和预处理
        print("\n📊 第一步：数据加载和预处理")
        sales_df = analyzer.load_integrated_data(start_date, end_date)
        enhanced_df = analyzer.create_enhanced_treatment_variables(sales_df)

        print(f"✅ 数据预处理完成，共 {len(enhanced_df)} 条记录，{len(enhanced_df.columns)} 个特征")

        # 2. 因果分析
        print("\n🎯 第二步：综合因果分析")
        causal_results = analyzer.analyze_promotion_effect_comprehensive(enhanced_df)

        # 3. 生成HTML报告
        print("\n📋 第三步：生成分析报告")
        report_file = analyzer.generate_html_report()

        print(f"\n🎉 分析完成！")
        print(f"📄 HTML报告: {report_file}")
        print("💡 请在浏览器中打开HTML文件查看详细结果")

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误:")
        print(f"错误信息: {e}")
        import traceback

        traceback.print_exc()

        print("\n🛠️ 建议检查:")
        print("1. 数据库连接是否正常")
        print("2. 数据时间范围是否有效")
        print("3. 是否有足够的数据进行分析")