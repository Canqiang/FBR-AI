"""
UMe 茶饮因果推断分析 HTML 报告生成器 - 新版
根据需求文档重新设计的报告生成器
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class UMeHTMLReportGenerator:
    """UMe 因果分析 HTML 报告生成器 - 新版"""

    def __init__(self):
        self.report_data = {}
        self.charts = {}

        # 集成行动推荐引擎
        try:
            from action_recommendation_engine import UMeActionRecommendationEngine
            self.action_engine = UMeActionRecommendationEngine()
            self.has_action_engine = True
        except ImportError:
            print("⚠️ 行动推荐引擎未找到，将使用基础建议")
            self.action_engine = None
            self.has_action_engine = False

    def generate_complete_report(self, analysis_results: Dict[str, Any], output_filename: str = None) -> str:
        """生成完整的HTML报告"""
        self.report_data = analysis_results

        # 生成行动推荐
        if self.has_action_engine:
            print("🎯 生成智能行动推荐...")
            causal_results = analysis_results.get('analysis_results', {})
            data_summary = analysis_results.get('data_summary', {})
            self.action_recommendations = self.action_engine.analyze_and_recommend(causal_results, data_summary)
        else:
            self.action_recommendations = None

        # 生成图表
        self._generate_all_charts()

        # 生成HTML内容
        html_content = self._build_html_report()

        # 保存文件
        if output_filename is None:
            start_date = analysis_results['analysis_period']['start']
            end_date = analysis_results['analysis_period']['end']
            output_filename = f"UMe_因果分析报告_{start_date}_{end_date}.html"

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML报告已生成: {output_filename}")
        return output_filename

    def _generate_all_charts(self):
        """生成所有图表"""
        analysis_results = self.report_data['analysis_results']

        # 1. 主要因素效应对比图
        self.charts['main_effects'] = self._create_main_effects_chart(analysis_results)

        # 2. 置信区间图
        self.charts['confidence_intervals'] = self._create_confidence_intervals_chart(analysis_results)

        # 3. 交互效应热力图
        if 'interactions' in analysis_results:
            self.charts['interactions'] = self._create_interactions_chart(analysis_results['interactions'])

        # 4. 异质性分析图
        if 'heterogeneity' in analysis_results:
            self.charts['heterogeneity'] = self._create_heterogeneity_chart(analysis_results['heterogeneity'])

        # 5. 天气产品影响图（按照实际category_name）
        self.charts['weather_products'] = self._create_weather_products_by_category()

        # 6. 时间趋势图（使用真实数据）
        self.charts['time_trends'] = self._create_time_trends_chart()

        # 7. 销售预测图（从forecast_results获取）
        self.charts['sales_forecast'] = self._create_sales_forecast_chart()

        # 8. 客户分析图（新增）
        self.charts['customer_analysis'] = self._create_customer_analysis_chart()

    def _create_weather_products_by_category(self) -> str:
        """按实际产品类别创建天气影响图"""
        enhanced_data = self.report_data.get('enhanced_data')

        if enhanced_data is None:
            return "<p>暂无数据</p>"

        try:
            # 定义产品类别
            category_columns = {
                'tea_drinks_orders': '茶饮类',
                'coffee_orders': '咖啡',
                'food_orders': '小食',
                'caffeine_free_orders': '无咖啡因',
                'new_product_orders': '新品'
            }

            # 天气条件
            weather_conditions = []
            category_data = {cat: [] for cat in category_columns.values()}

            # 分析不同天气下的销量
            weather_cols = {
                'is_hot': '高温天',
                'is_rainy': '雨天',
                'is_mild': '适宜天气',
                'is_cold': '寒冷天'
            }

            for weather_col, weather_name in weather_cols.items():
                if weather_col in enhanced_data.columns:
                    weather_data = enhanced_data[enhanced_data[weather_col] == 1]
                    if len(weather_data) > 0:
                        weather_conditions.append(weather_name)

                        for col, cat_name in category_columns.items():
                            if col in enhanced_data.columns:
                                avg_orders = weather_data[col].mean()
                                category_data[cat_name].append(avg_orders)

            # 添加整体平均
            if weather_conditions:
                weather_conditions.append('整体平均')
                for col, cat_name in category_columns.items():
                    if col in enhanced_data.columns:
                        avg_orders = enhanced_data[col].mean()
                        category_data[cat_name].append(avg_orders)

            # 创建图表
            fig = go.Figure()

            colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

            for i, (category, values) in enumerate(category_data.items()):
                if values:  # 只显示有数据的类别
                    fig.add_trace(go.Bar(
                        name=category,
                        x=weather_conditions,
                        y=values,
                        marker_color=colors[i % len(colors)]
                    ))

            fig.update_layout(
                title='不同天气条件下各产品类别的销量表现',
                xaxis_title='天气条件',
                yaxis_title='平均订单数',
                barmode='group',
                height=400,
                font=dict(size=12),
                plot_bgcolor='white'
            )

            return fig.to_html(include_plotlyjs='inline', div_id="weather_products_chart")

        except Exception as e:
            print(f"创建天气产品图失败: {e}")
            return "<p>创建图表时出错</p>"

    def _create_customer_analysis_chart(self) -> str:
        """创建客户分析图"""
        customer_data = self.report_data.get('customer_data')

        if customer_data is None or len(customer_data) == 0:
            return "<p>暂无客户数据</p>"

        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['客户类型分布', '消费等级分布', '活跃度分布', '客户价值分布'],
                specs=[[{'type': 'pie'}, {'type': 'pie'}],
                       [{'type': 'pie'}, {'type': 'bar'}]]
            )

            # 1. 客户类型分布
            customer_types = ['loyal', 'regular', 'potential', 'churned', 'dormant']
            type_labels = ['忠诚客户', '常规客户', '潜力客户', '流失客户', '休眠客户']
            type_counts = [customer_data[col].sum() for col in customer_types if col in customer_data.columns]

            if sum(type_counts) > 0:
                fig.add_trace(
                    go.Pie(labels=type_labels[:len(type_counts)], values=type_counts, name='客户类型'),
                    row=1, col=1
                )

            # 2. 消费等级分布
            spending_types = ['high_spending', 'medium_spending', 'low_spending']
            spending_labels = ['高消费', '中消费', '低消费']
            spending_counts = [customer_data[col].sum() for col in spending_types if col in customer_data.columns]

            if sum(spending_counts) > 0:
                fig.add_trace(
                    go.Pie(labels=spending_labels[:len(spending_counts)], values=spending_counts, name='消费等级'),
                    row=1, col=2
                )

            # 3. 活跃度分布
            activity_types = ['highly_active', 'moderately_active', 'low_active']
            activity_labels = ['高活跃', '中活跃', '低活跃']
            activity_counts = [customer_data[col].sum() for col in activity_types if col in customer_data.columns]

            if sum(activity_counts) > 0:
                fig.add_trace(
                    go.Pie(labels=activity_labels[:len(activity_counts)], values=activity_counts, name='活跃度'),
                    row=2, col=1
                )

            # 4. 客户价值分布（柱状图）
            value_types = ['high_value_customer', 'high_potential_customer', 'key_development_customer',
                          'regular_customer', 'general_value_customer']
            value_labels = ['高价值', '高潜力', '重点发展', '常规', '一般价值']
            value_counts = []
            value_labels_final = []

            for i, col in enumerate(value_types):
                if col in customer_data.columns:
                    count = customer_data[col].sum()
                    if count > 0:
                        value_counts.append(count)
                        value_labels_final.append(value_labels[i])

            if value_counts:
                fig.add_trace(
                    go.Bar(x=value_labels_final, y=value_counts, marker_color='#3498db'),
                    row=2, col=2
                )

            fig.update_layout(
                title='客户画像分析',
                showlegend=False,
                height=600
            )

            return fig.to_html(include_plotlyjs='inline', div_id="customer_analysis_chart")

        except Exception as e:
            print(f"创建客户分析图失败: {e}")
            return "<p>创建客户分析图时出错</p>"

    def _create_sales_forecast_chart(self) -> str:
        """创建销售预测图（从分析结果获取）"""
        forecast_results = self.report_data.get('forecast_results')

        if not forecast_results or 'forecast' not in forecast_results:
            return "<p>暂无销售预测数据</p>"

        try:
            forecast_df = forecast_results['forecast']
            summary = forecast_results['summary']

            # 创建图表
            fig = go.Figure()

            # 历史实际数据
            historical = forecast_df[forecast_df['y'].notna()]
            fig.add_trace(
                go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='lines+markers',
                    name='历史销售额',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=5),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>实际: $%{y:,.0f}<extra></extra>'
                )
            )

            # 预测数据
            future = forecast_df[forecast_df['ds'] > pd.to_datetime(summary['last_actual_date'])]
            if len(future) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=future['ds'],
                        y=future['yhat'],
                        mode='lines+markers',
                        name='预测销售额',
                        line=dict(color='#e74c3c', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond'),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>预测: $%{y:,.0f}<extra></extra>'
                    )
                )

                # 置信区间
                fig.add_trace(
                    go.Scatter(
                        x=future['ds'].tolist() + future['ds'].tolist()[::-1],
                        y=future['yhat_upper'].tolist() + future['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        line=dict(color='rgba(231, 76, 60, 0)'),
                        name='预测区间',
                        showlegend=True,
                        hoverinfo='skip'
                    )
                )

            # 添加分割线
            fig.add_vline(
                x=pd.to_datetime(summary['last_actual_date']),
                line_dash="dot",
                line_color="gray",
                annotation_text="预测开始",
                annotation_position="top"
            )

            # 更新布局
            fig.update_layout(
                title=f'销售额预测（未来{summary["forecast_days"]}天）',
                xaxis_title='日期',
                yaxis_title='销售额 ($)',
                height=500,
                font=dict(size=12),
                plot_bgcolor='white',
                hovermode='x unified'
            )

            # 添加摘要信息
            fig.add_annotation(
                text=f"预测总额: ${summary['total_forecast']:,.0f} | 日均: ${summary['avg_daily_forecast']:,.0f}",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=10, color="gray")
            )

            return fig.to_html(include_plotlyjs='inline', div_id="sales_forecast_chart")

        except Exception as e:
            print(f"创建销售预测图失败: {e}")
            return "<p>创建销售预测图时出错</p>"

    def _create_key_metrics_table(self) -> str:
        """创建关键指标表"""
        enhanced_data = self.report_data.get('enhanced_data')
        if enhanced_data is None:
            return "<p>暂无数据</p>"

        try:
            # 计算最近7天和14天的数据
            df = enhanced_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            max_date = df['date'].max()

            # 最近7天
            last_7d = df[df['date'] > max_date - timedelta(days=7)]
            # 前7天（用于对比）
            prev_7d = df[(df['date'] > max_date - timedelta(days=14)) &
                        (df['date'] <= max_date - timedelta(days=7))]

            # 计算指标
            metrics = []

            # 销售收入
            revenue_7d = last_7d['total_revenue'].sum()
            revenue_prev = prev_7d['total_revenue'].sum()
            revenue_change = ((revenue_7d - revenue_prev) / revenue_prev * 100) if revenue_prev > 0 else 0
            metrics.append({
                'metric': '销售收入',
                'value': f'${revenue_7d:,.0f}',
                'change': f'{revenue_change:+.1f}%',
                'status': '🟢' if revenue_change >= 0 else '🔴'
            })

            # 订单数
            orders_7d = last_7d['order_count'].sum()
            orders_prev = prev_7d['order_count'].sum()
            orders_change = ((orders_7d - orders_prev) / orders_prev * 100) if orders_prev > 0 else 0
            metrics.append({
                'metric': '订单数',
                'value': f'{orders_7d:,}',
                'change': f'{orders_change:+.1f}%',
                'status': '🟢' if orders_change >= 0 else '🔴'
            })

            # 客单价
            aov_7d = revenue_7d / orders_7d if orders_7d > 0 else 0
            aov_prev = revenue_prev / orders_prev if orders_prev > 0 else 0
            aov_change = ((aov_7d - aov_prev) / aov_prev * 100) if aov_prev > 0 else 0
            metrics.append({
                'metric': '客单价',
                'value': f'${aov_7d:.2f}',
                'change': f'{aov_change:+.1f}%',
                'status': '🟢' if aov_change >= 0 else '🔴'
            })

            # 客户数
            customers_7d = last_7d['unique_customers'].sum()
            customers_prev = prev_7d['unique_customers'].sum()
            customers_change = ((customers_7d - customers_prev) / customers_prev * 100) if customers_prev > 0 else 0
            metrics.append({
                'metric': '客户数',
                'value': f'{customers_7d:,}',
                'change': f'{customers_change:+.1f}%',
                'status': '🟢' if customers_change >= 0 else '🔴'
            })

            # 构建HTML表格
            html = """
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>最近7天</th>
                        <th>环比变化</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
            """

            for metric in metrics:
                html += f"""
                    <tr>
                        <td><strong>{metric['metric']}</strong></td>
                        <td>{metric['value']}</td>
                        <td>{metric['change']}</td>
                        <td>{metric['status']}</td>
                    </tr>
                """

            html += """
                </tbody>
            </table>
            """

            return html

        except Exception as e:
            print(f"创建关键指标表失败: {e}")
            return "<p>创建关键指标表时出错</p>"

    def _create_attribution_table(self) -> str:
        """创建根因分析表"""
        analysis_results = self.report_data.get('analysis_results', {})
        if not analysis_results:
            return "<p>暂无分析结果</p>"

        attributions = []

        # 分析各因素的影响
        for factor, result in analysis_results.items():
            if isinstance(result, dict) and 'ate' in result and 'error' not in result:
                ate = result.get('ate', 0)
                significant = result.get('significant', False)

                if significant and abs(ate) > 50:  # 只显示显著且影响较大的因素
                    attribution = {
                        'factor': factor,
                        'impact': ate,
                        'type': 'positive' if ate > 0 else 'negative'
                    }

                    # 添加解释
                    if factor == 'has_promotion' and ate < 0:
                        attribution['description'] = '促销活动造成负面影响，可能是促销过度或时机不当'
                        attribution['action'] = '调整促销策略，优化促销时机和力度'
                    elif factor == 'is_rainy' and ate < 0:
                        attribution['description'] = '雨天导致客流减少，影响销售'
                        attribution['action'] = '加强外卖服务，推出雨天专属优惠'
                    elif factor == 'is_weekend' and ate > 0:
                        attribution['description'] = '周末效应带来积极影响'
                        attribution['action'] = '加强周末营销，增加产能准备'
                    elif factor == 'is_hot' and ate > 0:
                        attribution['description'] = '高温天气促进冷饮销售'
                        attribution['action'] = '增加冷饮库存，推出消暑套餐'

                    attributions.append(attribution)

        # 排序：按影响程度
        attributions.sort(key=lambda x: abs(x['impact']), reverse=True)

        # 构建HTML
        if not attributions:
            return "<p>暂无显著的影响因素</p>"

        html = """
        <table class="attribution-table">
            <thead>
                <tr>
                    <th>影响因素</th>
                    <th>影响程度</th>
                    <th>原因分析</th>
                    <th>建议行动</th>
                </tr>
            </thead>
            <tbody>
        """

        for attr in attributions[:5]:  # 最多显示5个
            impact_class = 'positive' if attr['type'] == 'positive' else 'negative'
            html += f"""
                <tr>
                    <td><strong>{self._get_factor_name(attr['factor'])}</strong></td>
                    <td class="{impact_class}">${attr['impact']:+.0f}/天</td>
                    <td>{attr.get('description', '-')}</td>
                    <td>{attr.get('action', '-')}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def _get_factor_name(self, factor: str) -> str:
        """获取因素的中文名"""
        factor_names = {
            'has_promotion': '促销活动',
            'is_weekend': '周末',
            'is_holiday': '节假日',
            'is_hot': '高温天气',
            'is_rainy': '雨天',
            'is_cold': '寒冷天气'
        }
        return factor_names.get(factor, factor)

    def _create_intelligent_recommendations(self) -> str:
        """创建智能推荐部分"""
        # 获取预测数据
        forecast_results = self.report_data.get('forecast_results', {})
        forecast_df = forecast_results.get('forecast') if forecast_results else None

        # 获取分析结果
        analysis_results = self.report_data.get('analysis_results', {})

        recommendations = []

        # 基于预测的库存建议（预留接口）
        if forecast_df is not None:
            recommendations.append({
                'title': '销售预测提醒',
                'description': f"未来7天预计销售额${forecast_results['summary']['total_forecast']:,.0f}",
                'action': '请根据预测调整备货计划',
                'priority': 'high',
                'type': 'forecast'
            })

        # 基于因果分析的建议
        if self.action_recommendations:
            execution_plan = self.action_recommendations.get('execution_plan', {})
            critical_actions = execution_plan.get('priority_phases', {}).get('phase_1_critical', [])

            for action in critical_actions[:3]:  # 最多3个关键行动
                recommendations.append({
                    'title': action.get('name', '优化建议'),
                    'description': action.get('description', ''),
                    'action': '立即执行',
                    'priority': 'high',
                    'type': 'action'
                })

        # 基于天气的建议
        if 'is_hot' in analysis_results and analysis_results['is_hot'].get('significant'):
            ate = analysis_results['is_hot'].get('ate', 0)
            if ate > 0:
                recommendations.append({
                    'title': '高温天气营销机会',
                    'description': f'高温天气可带来${ate:.0f}/天的额外收入',
                    'action': '增加冷饮备货，推出清凉套餐',
                    'priority': 'medium',
                    'type': 'weather'
                })

        # 构建HTML
        if not recommendations:
            return "<p>暂无特定推荐</p>"

        html = '<div class="recommendations-container">'

        for rec in recommendations:
            priority_class = f"priority-{rec['priority']}"
            icon = {'high': '🔥', 'medium': '⭐', 'low': '💡'}.get(rec['priority'], '💡')

            html += f"""
            <div class="recommendation-card {priority_class}">
                <div class="rec-header">
                    <span class="rec-icon">{icon}</span>
                    <h4>{rec['title']}</h4>
                </div>
                <p class="rec-description">{rec['description']}</p>
                <button class="action-btn">{rec['action']}</button>
            </div>
            """

        html += '</div>'
        return html

    def _build_html_report(self) -> str:
        """构建完整的HTML报告"""
        # 获取基础数据
        data_summary = self.report_data.get('data_summary', {})
        analysis_results = self.report_data.get('analysis_results', {})

        # 计算关键指标
        key_findings = self._generate_key_findings(analysis_results)

        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMe 茶饮智能分析报告</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <header class="header">
            <div class="header-content">
                <h1><span class="logo">UMe</span> 茶饮智能分析报告</h1>
                <div class="report-meta">
                    <span>📅 分析期间: {self.report_data.get('analysis_period', {}).get('start', '')} ~ {self.report_data.get('analysis_period', {}).get('end', '')}</span>
                    <span>🏪 覆盖店铺: {data_summary.get('stores_count', 0)} 家</span>
                    <span>📊 样本量: {data_summary.get('total_records', 0)} 条记录</span>
                </div>
            </div>
        </header>

        <!-- 关键指标概览 -->
        <section class="key-metrics-section">
            <h2>📊 关键指标概览</h2>
            {self._create_key_metrics_table()}
        </section>

        <!-- 7天销售预测 -->
        <section class="forecast-section">
            <h2>🔮 销售预测与库存提醒</h2>
            <div class="chart-container">
                {self.charts.get('sales_forecast', '<p>暂无预测数据</p>')}
            </div>
        </section>

        <!-- 根因分析 -->
        <section class="attribution-section">
            <h2>🔍 根因分析</h2>
            {self._create_attribution_table()}
        </section>

        <!-- 智能推荐 -->
        <section class="recommendations-section">
            <h2>🎯 智能推荐</h2>
            {self._create_intelligent_recommendations()}
        </section>

        <!-- 详细分析 -->
        <section class="detailed-analysis">
            <h2>📈 详细分析</h2>
            
            <!-- 因果效应分析 -->
            <div class="analysis-group">
                <h3>各因素因果效应分析</h3>
                <div class="chart-container">
                    {self.charts.get('main_effects', '<p>暂无数据</p>')}
                </div>
                <div class="chart-container">
                    {self.charts.get('confidence_intervals', '<p>暂无数据</p>')}
                </div>
            </div>

            <!-- 交互效应分析 -->
            <div class="analysis-group">
                <h3>交互效应分析</h3>
                <div class="chart-container">
                    {self.charts.get('interactions', '<p>暂无交互效应数据</p>')}
                </div>
            </div>

            <!-- 异质性分析 -->
            <div class="analysis-group">
                <h3>异质性分析</h3>
                <div class="chart-container">
                    {self.charts.get('heterogeneity', '<p>暂无异质性数据</p>')}
                </div>
            </div>

            <!-- 天气与产品分析 -->
            <div class="analysis-group">
                <h3>天气与产品类别分析</h3>
                <div class="chart-container">
                    {self.charts.get('weather_products', '<p>暂无天气产品数据</p>')}
                </div>
            </div>

            <!-- 客户画像分析 -->
            <div class="analysis-group">
                <h3>客户画像分析</h3>
                <div class="chart-container">
                    {self.charts.get('customer_analysis', '<p>暂无客户数据</p>')}
                </div>
            </div>

            <!-- 时间趋势分析 -->
            <div class="analysis-group">
                <h3>时间趋势分析</h3>
                <div class="chart-container">
                    {self.charts.get('time_trends', '<p>暂无时间趋势数据</p>')}
                </div>
            </div>
        </section>

        <!-- 数据说明 -->
        <section class="data-notes">
            <h2>📝 数据说明</h2>
            <div class="notes-grid">
                <div class="note-card">
                    <h4>✅ 已使用数据</h4>
                    <ul>
                        <li>销售流水数据</li>
                        <li>客户画像数据</li>
                        <li>促销活动数据</li>
                        <li>天气数据</li>
                    </ul>
                </div>
                <div class="note-card">
                    <h4>⚠️ 预留接口</h4>
                    <ul>
                        <li>库存管理数据</li>
                        <li>客流量数据</li>
                        <li>供应链数据</li>
                        <li>实时POS数据</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- 页脚 -->
        <footer class="footer">
            <p>📄 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🔬 分析方法: EconML因果推断 + Prophet时序预测 + 机器学习</p>
            <p>⚠️ 建议定期更新分析以获得最新洞察</p>
        </footer>
    </div>

    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html_template

    # [保留原有的辅助方法]
    def _create_main_effects_chart(self, results: Dict[str, Any]) -> str:
        """创建主要因素效应对比图"""
        factors = []
        effects = []
        colors = []

        factor_names = {
            'has_promotion': '促销活动',
            'is_weekend': '周末效应',
            'is_holiday': '节假日效应',
            'is_hot': '高温天气',
            'is_rainy': '雨天天气'
        }

        color_map = {
            'has_promotion': '#1f77b4',
            'is_weekend': '#ff7f0e',
            'is_holiday': '#2ca02c',
            'is_hot': '#d62728',
            'is_rainy': '#9467bd'
        }

        for factor_key, result in results.items():
            if factor_key in factor_names and 'ate' in result and 'error' not in result:
                factors.append(factor_names[factor_key])
                effects.append(result['ate'])
                colors.append(color_map.get(factor_key, '#7f7f7f'))

        if not factors:
            return "<p>暂无有效的主要效应数据</p>"

        fig = go.Figure(go.Bar(
            x=factors,
            y=effects,
            text=[f"${v:+.0f}" for v in effects],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>平均效应: $%{y:+.0f}<extra></extra>'
        ))

        fig.update_layout(
            title='各因素对营收的因果效应',
            xaxis_title='影响因素',
            yaxis_title='平均因果效应 ($)',
            showlegend=False,
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True, zerolinecolor='red', zerolinewidth=2)
        )

        return fig.to_html(include_plotlyjs='inline', div_id="main_effects_chart")

    def _create_time_trends_chart(self) -> str:
        """创建时间趋势图（使用真实数据）"""
        enhanced_data = self.report_data.get('enhanced_data')
        if enhanced_data is None:
            return "<p>暂无时间趋势数据</p>"

        try:
            # 确保日期格式正确
            enhanced_data = enhanced_data.copy()
            enhanced_data['date'] = pd.to_datetime(enhanced_data['date'])

            # 按日期聚合数据
            daily_data = enhanced_data.groupby('date').agg({
                'total_revenue': 'sum',
                'has_promotion': 'mean',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()

            # 计算7天移动平均
            daily_data['revenue_ma7'] = daily_data['total_revenue'].rolling(window=7, min_periods=1).mean()

            # 创建图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 日营收柱状图
            fig.add_trace(
                go.Bar(
                    x=daily_data['date'],
                    y=daily_data['total_revenue'],
                    name='日营收',
                    marker_color='rgba(52, 152, 219, 0.6)',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>营收: $%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False,
            )

            # 7天移动平均线
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'],
                    y=daily_data['revenue_ma7'],
                    name='7天移动平均',
                    line=dict(color='#2c3e50', width=3),
                    mode='lines',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>7天平均: $%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False,
            )

            # 促销标记
            promotion_days = daily_data[daily_data['has_promotion'] > 0.5]
            if len(promotion_days) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=promotion_days['date'],
                        y=promotion_days['total_revenue'],
                        mode='markers',
                        name='促销日',
                        marker=dict(
                            symbol='star',
                            size=10,
                            color='#f39c12',
                            line=dict(color='#e67e22', width=2)
                        ),
                        hovertemplate='<b>促销日</b><br>%{x|%Y-%m-%d}<br>营收: $%{y:,.0f}<extra></extra>'
                    ),
                    secondary_y=False,
                )

            # 更新布局
            fig.update_yaxes(title_text="营收 ($)", secondary_y=False, gridcolor='lightgray')
            fig.update_xaxes(title_text="日期", rangeslider_visible=True)

            fig.update_layout(
                title='营收时间趋势',
                height=500,
                font=dict(size=12),
                plot_bgcolor='white',
                hovermode='x unified'
            )

            return fig.to_html(include_plotlyjs='inline', div_id="time_trends_chart")

        except Exception as e:
            print(f"创建时间趋势图失败: {e}")
            return "<p>创建时间趋势图时出错</p>"

    def _create_confidence_intervals_chart(self, results: Dict[str, Any]) -> str:
        """创建置信区间图"""
        factors = []
        ates = []
        ci_lowers = []
        ci_uppers = []
        significances = []

        factor_names = {
            'has_promotion': '促销活动',
            'is_weekend': '周末效应',
            'is_holiday': '节假日效应',
            'is_hot': '高温天气',
            'is_rainy': '雨天天气'
        }

        for factor_key, result in results.items():
            if factor_key in factor_names and 'ate' in result and 'error' not in result:
                factors.append(factor_names[factor_key])
                ates.append(result['ate'])
                ci_lowers.append(result.get('ci_lower', result['ate'] - abs(result['ate']) * 0.2))
                ci_uppers.append(result.get('ci_upper', result['ate'] + abs(result['ate']) * 0.2))
                significances.append(result.get('significant', False))

        if not factors:
            return "<p>暂无置信区间数据</p>"

        fig = go.Figure()

        # 添加误差条
        for i, (factor, ate, ci_lower, ci_upper, significant) in enumerate(zip(factors, ates, ci_lowers, ci_uppers, significances)):
            color = '#2ca02c' if significant else '#ff7f0e'

            fig.add_trace(go.Scatter(
                x=[ate],
                y=[i],
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[ci_upper - ate],
                    arrayminus=[ate - ci_lower],
                    color=color,
                    thickness=3
                ),
                mode='markers',
                marker=dict(size=10, color=color),
                name=factor,
                hovertemplate=f'<b>{factor}</b><br>' +
                             f'ATE: ${ate:+.0f}<br>' +
                             f'95% CI: [${ci_lower:.0f}, ${ci_upper:.0f}]<br>' +
                             f'状态: {"显著" if significant else "不显著"}<extra></extra>'
            ))

        fig.update_layout(
            title='效应的置信区间（95%置信水平）',
            xaxis_title='平均因果效应 ($)',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(factors))),
                ticktext=factors
            ),
            showlegend=False,
            height=max(300, len(factors) * 50),
            font=dict(size=12),
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', zeroline=True, zerolinecolor='red', zerolinewidth=2),
            yaxis_title=""
        )

        # 添加零线
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="无效应线")

        return fig.to_html(include_plotlyjs='inline', div_id="confidence_intervals_chart")

    def _create_interactions_chart(self, interactions: Dict[str, Any]) -> str:
        """创建交互效应图"""
        if not interactions:
            return "<p>暂无交互效应数据</p>"

        interaction_names = []
        interaction_effects = []

        name_mapping = {
            'is_rainy_x_has_promotion': '雨天 × 促销',
            'is_hot_x_has_promotion': '高温 × 促销',
            'is_weekend_x_has_promotion': '周末 × 促销',
            'is_holiday_x_is_weekend': '节假日 × 周末'
        }

        for key, result in interactions.items():
            if 'error' not in result and 'interaction_effect' in result:
                interaction_names.append(name_mapping.get(key, key))
                interaction_effects.append(result['interaction_effect'])

        if not interaction_names:
            return "<p>暂无有效的交互效应数据</p>"

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='交互效应',
            x=interaction_names,
            y=interaction_effects,
            text=[f"${v:+.0f}" for v in interaction_effects],
            textposition='auto',
            marker_color='#e74c3c',
            hovertemplate='<b>%{x}</b><br>交互效应: $%{y:+.0f}<extra></extra>'
        ))

        fig.update_layout(
            title='交互效应分析',
            xaxis_title='因素组合',
            yaxis_title='交互效应 ($)',
            showlegend=False,
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True, zerolinecolor='gray', zerolinewidth=1)
        )

        return fig.to_html(include_plotlyjs='inline', div_id="interactions_chart")

    def _create_heterogeneity_chart(self, heterogeneity: Dict[str, Any]) -> str:
        """创建异质性分析图"""
        if not heterogeneity:
            return "<p>暂无异质性分析数据</p>"

        # 计算子图数量
        subplot_count = 0
        if 'promotion_by_store' in heterogeneity:
            subplot_count += 1
        if 'promotion_by_weather' in heterogeneity:
            subplot_count += 1
        if 'promotion_by_category' in heterogeneity:
            subplot_count += 1

        if subplot_count == 0:
            return "<p>暂无异质性分析数据</p>"

        # 创建子图
        fig = make_subplots(
            rows=1, cols=subplot_count,
            subplot_titles=['店铺差异', '天气条件差异', '产品类别差异'][:subplot_count],
            horizontal_spacing=0.1
        )

        col = 1

        # 店铺异质性
        if 'promotion_by_store' in heterogeneity:
            store_data = heterogeneity['promotion_by_store']
            if store_data:
                stores = list(store_data.keys())[:10]
                effects = [store_data[store]['effect'] for store in stores]

                fig.add_trace(
                    go.Bar(x=stores, y=effects, name='店铺效应',
                          text=[f"${v:+.0f}" for v in effects],
                          textposition='auto',
                          marker_color='#3498db'),
                    row=1, col=col
                )
                col += 1

        # 天气异质性
        if 'promotion_by_weather' in heterogeneity:
            weather_data = heterogeneity['promotion_by_weather']
            if weather_data:
                weather_names = {'is_hot': '高温天', 'is_rainy': '雨天', 'is_mild': '适宜天气'}
                conditions = []
                effects = []

                for condition, data in weather_data.items():
                    conditions.append(weather_names.get(condition, condition))
                    effects.append(data['effect'])

                fig.add_trace(
                    go.Bar(x=conditions, y=effects, name='天气效应',
                          text=[f"${v:+.0f}" for v in effects],
                          textposition='auto',
                          marker_color='#e67e22'),
                    row=1, col=col
                )
                col += 1

        # 产品类别异质性
        if 'promotion_by_category' in heterogeneity:
            category_data = heterogeneity['promotion_by_category']
            if category_data:
                categories = list(category_data.keys())
                lifts = [data['lift'] * 100 for data in category_data.values()]

                fig.add_trace(
                    go.Bar(x=categories, y=lifts, name='类别提升',
                          text=[f"{v:+.1f}%" for v in lifts],
                          textposition='auto',
                          marker_color='#9b59b6'),
                    row=1, col=col
                )

        fig.update_layout(
            title='异质性分析：不同维度的效果差异',
            showlegend=False,
            height=400,
            font=dict(size=12)
        )

        return fig.to_html(include_plotlyjs='inline', div_id="heterogeneity_chart")

    def _generate_key_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成关键发现"""
        findings = {
            'strongest_positive': None,
            'strongest_negative': None,
            'most_significant': None,
            'total_factors': 0
        }

        max_positive = float('-inf')
        max_negative = float('inf')
        most_significant_factor = None
        most_significant_effect = 0

        factor_names = {
            'has_promotion': '促销活动',
            'is_weekend': '周末效应',
            'is_holiday': '节假日效应',
            'is_hot': '高温天气',
            'is_rainy': '雨天天气'
        }

        for factor_key, result in results.items():
            if factor_key in factor_names and 'ate' in result and 'error' not in result:
                findings['total_factors'] += 1
                ate = result['ate']
                significant = result.get('significant', False)

                if ate > max_positive:
                    max_positive = ate
                    findings['strongest_positive'] = {
                        'name': factor_names[factor_key],
                        'effect': ate,
                        'significant': significant
                    }

                if ate < max_negative:
                    max_negative = ate
                    findings['strongest_negative'] = {
                        'name': factor_names[factor_key],
                        'effect': ate,
                        'significant': significant
                    }

                if significant and abs(ate) > abs(most_significant_effect):
                    most_significant_effect = ate
                    most_significant_factor = factor_names[factor_key]

        if most_significant_factor:
            findings['most_significant'] = {
                'name': most_significant_factor,
                'effect': most_significant_effect
            }

        return findings

    def _get_css_styles(self) -> str:
        """获取CSS样式"""
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
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .logo {
            font-weight: bold;
            font-size: 1.2em;
            background: rgba(255,255,255,0.2);
            padding: 5px 10px;
            border-radius: 5px;
        }
        
        .report-meta {
            margin-top: 15px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .report-meta span {
            background: rgba(255,255,255,0.1);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #34495e;
            margin-bottom: 15px;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #34495e;
        }
        
        .metrics-table td:nth-child(2) {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .attribution-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .attribution-table th,
        .attribution-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .attribution-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #34495e;
        }
        
        .attribution-table .positive {
            color: #27ae60;
            font-weight: bold;
        }
        
        .attribution-table .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .recommendations-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .recommendation-card {
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            background: white;
            border: 1px solid #e0e0e0;
        }
        
        .recommendation-card.priority-high {
            border-left: 4px solid #e74c3c;
        }
        
        .recommendation-card.priority-medium {
            border-left: 4px solid #f39c12;
        }
        
        .recommendation-card.priority-low {
            border-left: 4px solid #3498db;
        }
        
        .rec-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .rec-icon {
            font-size: 1.5em;
        }
        
        .rec-header h4 {
            margin: 0;
            color: #2c3e50;
        }
        
        .rec-description {
            color: #555;
            margin-bottom: 15px;
        }
        
        .action-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
        }
        
        .action-btn:hover {
            background: #2980b9;
        }
        
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: white;
        }
        
        .analysis-group {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .data-notes {
            background: #f8f9fa;
        }
        
        .notes-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .note-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .note-card h4 {
            color: #2980b9;
            margin-bottom: 10px;
        }
        
        .note-card ul {
            margin-left: 20px;
        }
        
        .note-card li {
            margin: 5px 0;
            color: #555;
        }
        
        .footer {
            background: #34495e;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .footer p {
            margin: 5px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .report-meta {
                flex-direction: column;
                gap: 10px;
            }
            
            section {
                padding: 20px;
            }
            
            .recommendations-container,
            .notes-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _get_javascript(self) -> str:
        """获取JavaScript代码"""
        return """
        // 页面加载完成后的初始化
        document.addEventListener('DOMContentLoaded', function() {
            // 为操作按钮添加事件监听
            const actionBtns = document.querySelectorAll('.action-btn');
            actionBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    alert('功能开发中，敬请期待！');
                });
            });
            
            // 添加图表容器的加载动画
            const chartContainers = document.querySelectorAll('.chart-container');
            chartContainers.forEach(container => {
                container.style.opacity = '0';
                container.style.transform = 'translateY(20px)';
                container.style.transition = 'all 0.5s ease';
                
                setTimeout(() => {
                    container.style.opacity = '1';
                    container.style.transform = 'translateY(0)';
                }, 100);
            });
        });
        """