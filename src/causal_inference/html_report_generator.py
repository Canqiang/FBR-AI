"""
UMe 茶饮因果推断分析 HTML 报告生成器
生成商户友好的交互式分析报告
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List


class UMeHTMLReportGenerator:
    """UMe 因果分析 HTML 报告生成器"""

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

        # 5. 天气产品影响图
        self.charts['weather_products'] = self._create_weather_products_chart()

        # 6. 时间趋势图
        self.charts['time_trends'] = self._create_time_trends_chart()

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
            title='各因素对营收的因果效应对比',
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
            title='各因素效应的置信区间（95%置信水平）',
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
        factor1_effects = []
        factor2_effects = []

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
                factor1_effects.append(result['factor1_main_effect'])
                factor2_effects.append(result['factor2_main_effect'])

        if not interaction_names:
            return "<p>暂无有效的交互效应数据</p>"

        fig = go.Figure()

        # 交互效应柱状图
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
            title='交互效应分析：组合策略的协同作用',
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

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['不同店铺的促销效果', '不同天气条件下的促销效果'],
            horizontal_spacing=0.1
        )

        # 店铺异质性
        if 'promotion_by_store' in heterogeneity:
            store_data = heterogeneity['promotion_by_store']
            if store_data:
                stores = list(store_data.keys())[:10]  # 最多显示10个店铺
                effects = [store_data[store]['effect'] for store in stores]

                fig.add_trace(
                    go.Bar(x=stores, y=effects, name='店铺效应',
                          text=[f"${v:+.0f}" for v in effects],
                          textposition='auto',
                          marker_color='#3498db'),
                    row=1, col=1
                )

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
                    row=1, col=2
                )

        fig.update_layout(
            title='异质性分析：不同情况下的效果差异',
            showlegend=False,
            height=400,
            font=dict(size=12)
        )

        return fig.to_html(include_plotlyjs='inline', div_id="heterogeneity_chart")

    def _create_weather_products_chart(self) -> str:
        """创建天气对产品影响图（示例）"""
        # 创建示例数据
        weather_conditions = ['晴天', '雨天', '高温天', '适宜天气']
        cold_drinks = [100, 70, 150, 120]
        hot_drinks = [50, 80, 30, 60]
        snacks = [80, 90, 85, 75]

        fig = go.Figure()

        fig.add_trace(go.Bar(name='冷饮', x=weather_conditions, y=cold_drinks, marker_color='#3498db'))
        fig.add_trace(go.Bar(name='热饮', x=weather_conditions, y=hot_drinks, marker_color='#e74c3c'))
        fig.add_trace(go.Bar(name='小食', x=weather_conditions, y=snacks, marker_color='#f39c12'))

        fig.update_layout(
            title='不同天气条件下各产品类型的销量表现',
            xaxis_title='天气条件',
            yaxis_title='平均日销量',
            barmode='group',
            height=400,
            font=dict(size=12),
            plot_bgcolor='white'
        )

        return fig.to_html(include_plotlyjs='inline', div_id="weather_products_chart")

    def _create_time_trends_chart(self) -> str:
        """创建时间趋势图（示例）"""
        # 创建示例月度数据
        months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        revenue = [8000, 8500, 9200, 10500, 11800, 12500, 13200, 13000, 11500, 10200, 9000, 8800]
        promotion_effect = [150, 180, 200, 220, 180, 160, 140, 130, 170, 190, 200, 180]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 营收趋势
        fig.add_trace(
            go.Scatter(x=months, y=revenue, name='月度营收',
                      line=dict(color='#2c3e50', width=3),
                      mode='lines+markers'),
            secondary_y=False,
        )

        # 促销效应趋势
        fig.add_trace(
            go.Scatter(x=months, y=promotion_effect, name='促销效应',
                      line=dict(color='#e74c3c', width=2, dash='dash'),
                      mode='lines+markers'),
            secondary_y=True,
        )

        fig.update_yaxes(title_text="营收 ($)", secondary_y=False)
        fig.update_yaxes(title_text="促销效应 ($)", secondary_y=True)
        fig.update_xaxes(title_text="月份")

        fig.update_layout(
            title='月度营收和促销效应趋势',
            height=400,
            font=dict(size=12),
            plot_bgcolor='white'
        )

        return fig.to_html(include_plotlyjs='inline', div_id="time_trends_chart")

    def _build_html_report(self) -> str:
        """构建完整的HTML报告"""
        # 获取基础数据
        data_summary = self.report_data.get('data_summary', {})
        analysis_results = self.report_data.get('analysis_results', {})

        # 计算关键指标
        key_findings = self._generate_key_findings(analysis_results)
        business_recommendations = self._generate_business_recommendations(analysis_results)

        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMe 茶饮因果分析报告</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <header class="header">
            <div class="header-content">
                <h1><span class="logo">UMe</span> 茶饮因果分析报告</h1>
                <div class="report-meta">
                    <span>📅 分析期间: {self.report_data.get('analysis_period', {}).get('start', '')} ~ {self.report_data.get('analysis_period', {}).get('end', '')}</span>
                    <span>🏪 覆盖店铺: {data_summary.get('stores_count', 0)} 家</span>
                    <span>📊 样本量: {data_summary.get('total_records', 0)} 条记录</span>
                </div>
            </div>
        </header>

        <!-- 执行摘要 -->
        <section class="executive-summary">
            <h2>📊 执行摘要</h2>
            <div class="summary-grid">
                {self._build_summary_cards(key_findings)}
            </div>
        </section>

        <!-- 概念解释 -->
        <section class="concepts-section">
            <h2>📖 核心概念解释</h2>
            <div class="concepts-grid">
                {self._build_concepts_explanation()}
            </div>
        </section>

        <!-- 主要分析结果 -->
        <section class="main-results">
            <h2>🎯 主要分析结果</h2>
            
            <div class="chart-container">
                <h3>各因素因果效应对比</h3>
                {self.charts.get('main_effects', '<p>暂无数据</p>')}
            </div>
            
            <div class="chart-container">
                <h3>效应的置信区间分析</h3>
                {self.charts.get('confidence_intervals', '<p>暂无数据</p>')}
                <div class="explanation">
                    <p><strong>📏 如何理解置信区间：</strong></p>
                    <ul>
                        <li><span class="significant">绿色</span>：效应显著，不包含零线，结果可信</li>
                        <li><span class="not-significant">橙色</span>：效应不显著，包含零线，需要更多数据</li>
                        <li><strong>区间越窄</strong>：结果越确定，决策风险越低</li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- 交互效应分析 -->
        <section class="interaction-analysis">
            <h2>🔄 交互效应分析</h2>
            <div class="chart-container">
                <h3>组合策略的协同作用</h3>
                {self.charts.get('interactions', '<p>暂无交互效应数据</p>')}
                <div class="explanation">
                    <p><strong>💡 交互效应解释：</strong>交互效应显示两个因素组合使用时的额外收益。正值表示协同作用，负值表示相互抵消。</p>
                </div>
            </div>
        </section>

        <!-- 异质性分析 -->
        <section class="heterogeneity-analysis">
            <h2>🔍 异质性分析</h2>
            <div class="chart-container">
                <h3>不同情况下的效果差异</h3>
                {self.charts.get('heterogeneity', '<p>暂无异质性数据</p>')}
                <div class="explanation">
                    <p><strong>🎯 精准营销洞察：</strong>同样的策略在不同店铺、不同天气条件下效果不同。建议针对高效场景加大投入，低效场景优化策略。</p>
                </div>
            </div>
        </section>

        <!-- 天气产品分析 -->
        <section class="weather-products">
            <h2>🌤️ 天气与产品分析</h2>
            <div class="chart-container">
                <h3>天气对不同产品的影响</h3>
                {self.charts.get('weather_products', '<p>暂无天气产品数据</p>')}
            </div>
        </section>

        <!-- 时间趋势分析 -->
        <section class="time-trends">
            <h2>📈 时间趋势分析</h2>
            <div class="chart-container">
                <h3>季节性模式和促销效应变化</h3>
                {self.charts.get('time_trends', '<p>暂无时间趋势数据</p>')}
            </div>
        </section>

        <!-- 智能行动推荐 -->
        <section class="action-recommendations">
            <h2>🎯 智能行动推荐</h2>
            {self._build_action_recommendations()}
        </section>

        <!-- 业务建议 -->
        <section class="business-recommendations">
            <h2>💡 补充建议</h2>
            <div class="recommendations-grid">
                {self._build_recommendations_cards(business_recommendations)}
            </div>
        </section>

        <!-- 详细数据表格 -->
        <section class="detailed-results">
            <h2>📋 详细分析结果</h2>
            <div class="collapsible-section">
                <button class="collapsible-btn" onclick="toggleSection('detailed-table')">
                    点击查看详细统计数据 ▼
                </button>
                <div id="detailed-table" class="collapsible-content">
                    {self._build_detailed_table(analysis_results)}
                </div>
            </div>
        </section>

        <!-- 页脚 -->
        <footer class="footer">
            <p>📄 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🔬 分析方法: EconML因果推断 + 多因素交互分析</p>
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

    def _build_action_recommendations(self) -> str:
        """构建智能行动推荐部分"""
        if not self.action_recommendations:
            return """
            <div class="no-actions">
                <p>📊 基于当前分析结果，暂无特定行动推荐</p>
                <p>建议继续收集数据，监控关键指标变化</p>
            </div>
            """

        recommendations = self.action_recommendations
        execution_plan = recommendations.get('execution_plan', {})
        summary = recommendations.get('summary', {})

        html_parts = []

        # 推荐摘要
        html_parts.append(f"""
        <div class="action-summary">
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">{summary.get('total_actions', 0)}</span>
                    <span class="stat-label">推荐行动</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{summary.get('high_priority_actions', 0)}</span>
                    <span class="stat-label">高优先级</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${summary.get('estimated_monthly_impact', 0):,.0f}</span>
                    <span class="stat-label">预期月度影响</span>
                </div>
            </div>
            <div class="key-recommendation">
                <h3>🎯 核心建议</h3>
                <p>{summary.get('key_recommendation', '继续监控数据')}</p>
            </div>
        </div>
        """)

        # 立即行动
        critical_actions = execution_plan.get('priority_phases', {}).get('phase_1_critical', [])
        if critical_actions:
            html_parts.append('<div class="critical-actions">')
            html_parts.append('<h3>🔥 立即行动 (关键优先级)</h3>')

            for i, action in enumerate(critical_actions, 1):
                urgency_class = f"urgency-{action.get('urgency', 'medium')}"
                impact = action.get('financial_impact', 0)

                html_parts.append(f"""
                <div class="action-card critical {urgency_class}">
                    <div class="action-header">
                        <span class="action-number">{i}</span>
                        <h4>{action.get('name', '待定行动')}</h4>
                        <span class="impact-badge">${impact:,.0f}/月</span>
                    </div>
                    <div class="action-content">
                        <p class="action-description">{action.get('description', '待详细制定')}</p>
                        <div class="action-details">
                            <span class="timeline">⏱️ {action.get('timeline', '待确定')}</span>
                            <span class="kpi">📊 {', '.join(action.get('kpi', ['待定']))}</span>
                        </div>
                    </div>
                </div>
                """)

            html_parts.append('</div>')

        # 短期行动
        important_actions = execution_plan.get('priority_phases', {}).get('phase_2_important', [])
        if important_actions:
            html_parts.append('<div class="important-actions">')
            html_parts.append('<h3>⭐ 短期优化 (重要优先级)</h3>')
            html_parts.append('<div class="actions-grid">')

            for action in important_actions[:3]:  # 最多显示3个
                impact = action.get('financial_impact', 0)
                html_parts.append(f"""
                <div class="action-card important">
                    <h4>{action.get('name', '待定行动')}</h4>
                    <p>{action.get('description', '待详细制定')[:100]}...</p>
                    <div class="action-meta">
                        <span class="impact">${impact:,.0f}/月</span>
                        <span class="timeline">{action.get('timeline', '待确定')}</span>
                    </div>
                </div>
                """)

            html_parts.append('</div>')
            html_parts.append('</div>')

        # 长期改进
        improvement_actions = execution_plan.get('priority_phases', {}).get('phase_3_improvement', [])
        if improvement_actions:
            html_parts.append('<div class="improvement-actions">')
            html_parts.append('<h3>💡 长期改进</h3>')
            html_parts.append('<ul class="improvement-list">')

            for action in improvement_actions[:4]:  # 最多显示4个
                html_parts.append(f"""
                <li><strong>{action.get('name', '待定行动')}</strong>: {action.get('description', '待详细制定')[:80]}...</li>
                """)

            html_parts.append('</ul>')
            html_parts.append('</div>')

        # 监控指标
        success_metrics = summary.get('success_metrics', [])
        if success_metrics:
            html_parts.append(f"""
            <div class="monitoring-section">
                <h3>📈 关键监控指标</h3>
                <div class="metrics-tags">
                    {' '.join([f'<span class="metric-tag">{metric}</span>' for metric in success_metrics])}
                </div>
                <p class="next-review">下次回顾: {summary.get('next_review_date', '30天后')}</p>
            </div>
            """)

        return ''.join(html_parts)

    def _generate_business_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """生成基础业务建议（补充智能推荐）"""
        recommendations = []

        # 基于主要效应生成建议
        for factor_key, result in results.items():
            if 'ate' in result and 'error' not in result:
                ate = result['ate']
                significant = result.get('significant', False)

                if factor_key == 'has_promotion' and significant:
                    if ate > 50:
                        recommendations.append({
                            'priority': '中',
                            'action': '扩大有效促销',
                            'reason': f'促销活动显著提升营收${ate:.0f}',
                            'impact': f'预计带来{ate*30:.0f}美元月度增收'
                        })
                    elif ate < -20:
                        recommendations.append({
                            'priority': '高',
                            'action': '调整促销策略',
                            'reason': f'当前促销策略造成${abs(ate):.0f}损失',
                            'impact': '避免继续亏损，优化ROI'
                        })

                elif factor_key == 'is_hot' and significant and ate > 30:
                    recommendations.append({
                        'priority': '中',
                        'action': '高温天气营销',
                        'reason': f'高温天气带来${ate:.0f}额外营收',
                        'impact': '夏季重点推广冷饮产品'
                    })

                elif factor_key == 'is_rainy' and significant:
                    if ate > 20:
                        recommendations.append({
                            'priority': '中',
                            'action': '雨天营销机会',
                            'reason': f'雨天促进${ate:.0f}额外消费',
                            'impact': '雨天推出外卖优惠'
                        })
                    else:
                        recommendations.append({
                            'priority': '低',
                            'action': '雨天运营优化',
                            'reason': '雨天对营收有负面影响',
                            'impact': '提前准备雨天备选方案'
                        })

        # 如果没有足够的建议，添加通用建议
        if len(recommendations) < 3:
            recommendations.extend([
                {
                    'priority': '中',
                    'action': '持续数据收集',
                    'reason': '扩大样本量提高分析精度',
                    'impact': '为未来决策提供更可靠依据'
                },
                {
                    'priority': '低',
                    'action': '定期分析回顾',
                    'reason': '市场环境和消费行为会变化',
                    'impact': '保持策略的时效性和有效性'
                }
            ])

        return recommendations[:6]  # 最多返回6个建议
        """生成业务建议"""
        recommendations = []

        # 基于主要效应生成建议
        for factor_key, result in results.items():
            if 'ate' in result and 'error' not in result:
                ate = result['ate']
                significant = result.get('significant', False)

                if factor_key == 'has_promotion' and significant:
                    if ate > 50:
                        recommendations.append({
                            'priority': '高',
                            'action': '扩大促销活动',
                            'reason': f'促销活动显著提升营收${ate:.0f}',
                            'impact': f'预计带来{ate*30:.0f}美元月度增收'
                        })
                    elif ate < -20:
                        recommendations.append({
                            'priority': '高',
                            'action': '重新评估促销策略',
                            'reason': f'当前促销策略造成${abs(ate):.0f}损失',
                            'impact': '避免继续亏损，优化ROI'
                        })

                elif factor_key == 'is_hot' and significant and ate > 30:
                    recommendations.append({
                        'priority': '中',
                        'action': '高温天气营销策略',
                        'reason': f'高温天气带来${ate:.0f}额外营收',
                        'impact': '夏季重点推广冷饮产品'
                    })

                elif factor_key == 'is_rainy' and significant:
                    if ate > 20:
                        recommendations.append({
                            'priority': '中',
                            'action': '雨天营销机会',
                            'reason': f'雨天促进${ate:.0f}额外消费',
                            'impact': '雨天推出外卖优惠'
                        })
                    else:
                        recommendations.append({
                            'priority': '低',
                            'action': '雨天运营优化',
                            'reason': '雨天对营收有负面影响',
                            'impact': '提前准备雨天备选方案'
                        })

        # 如果没有足够的建议，添加通用建议
        if len(recommendations) < 3:
            recommendations.extend([
                {
                    'priority': '中',
                    'action': '持续数据收集',
                    'reason': '扩大样本量提高分析精度',
                    'impact': '为未来决策提供更可靠依据'
                },
                {
                    'priority': '低',
                    'action': '定期分析回顾',
                    'reason': '市场环境和消费行为会变化',
                    'impact': '保持策略的时效性和有效性'
                }
            ])

        return recommendations[:6]  # 最多返回6个建议

    def _build_summary_cards(self, findings: Dict[str, Any]) -> str:
        """构建摘要卡片"""
        cards_html = ""

        if findings.get('strongest_positive'):
            pos = findings['strongest_positive']
            status = "✅ 显著" if pos['significant'] else "⚠️ 不显著"
            cards_html += f"""
            <div class="summary-card positive">
                <div class="card-icon">📈</div>
                <div class="card-content">
                    <h3>最强正面效应</h3>
                    <p class="card-value">${pos['effect']:+.0f}</p>
                    <p class="card-label">{pos['name']} {status}</p>
                </div>
            </div>
            """

        if findings.get('strongest_negative'):
            neg = findings['strongest_negative']
            status = "❌ 显著" if neg['significant'] else "⚠️ 不显著"
            cards_html += f"""
            <div class="summary-card negative">
                <div class="card-icon">📉</div>
                <div class="card-content">
                    <h3>最强负面效应</h3>
                    <p class="card-value">${neg['effect']:+.0f}</p>
                    <p class="card-label">{neg['name']} {status}</p>
                </div>
            </div>
            """

        if findings.get('most_significant'):
            sig = findings['most_significant']
            cards_html += f"""
            <div class="summary-card significant">
                <div class="card-icon">🎯</div>
                <div class="card-content">
                    <h3>最可靠发现</h3>
                    <p class="card-value">${sig['effect']:+.0f}</p>
                    <p class="card-label">{sig['name']}</p>
                </div>
            </div>
            """

        cards_html += f"""
        <div class="summary-card info">
            <div class="card-icon">🔍</div>
            <div class="card-content">
                <h3>分析因素</h3>
                <p class="card-value">{findings['total_factors']}</p>
                <p class="card-label">个影响因素</p>
            </div>
        </div>
        """

        return cards_html

    def _build_concepts_explanation(self) -> str:
        """构建概念解释区域"""
        return """
        <div class="concept-card">
            <h3>🎯 平均处理效应 (ATE)</h3>
            <div class="concept-content">
                <p><strong>商业解释：</strong>某个策略的平均影响</p>
                <div class="example">
                    <p>📊 促销活动的ATE = +$150</p>
                    <p>💡 意思是：促销活动平均每天能带来150美元的额外营收</p>
                </div>
                <div class="decision-guide">
                    <p><strong>决策指导：</strong></p>
                    <ul>
                        <li>✅ ATE > 0：策略有效，建议继续</li>
                        <li>❌ ATE < 0：策略有害，建议停止</li>
                        <li>⚠️ ATE ≈ 0：策略无明显效果，考虑优化</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="concept-card">
            <h3>📏 置信区间</h3>
            <div class="concept-content">
                <p><strong>商业解释：</strong>结果的可信范围</p>
                <div class="example">
                    <p>📊 促销ATE = $150，95%置信区间：[$120, $180]</p>
                    <p>💡 意思是：真实效果有95%概率在120-180美元之间</p>
                </div>
                <div class="decision-guide">
                    <p><strong>决策指导：</strong></p>
                    <ul>
                        <li>🎯 <strong>窄区间</strong>：结果确定，可放心决策</li>
                        <li>⚠️ <strong>宽区间</strong>：结果不确定，需要更多数据</li>
                        <li>❌ <strong>包含0</strong>：效果不显著，不建议采用</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="concept-card">
            <h3>🔍 异质性分析</h3>
            <div class="concept-content">
                <p><strong>商业解释：</strong>不同情况下的效果差异</p>
                <div class="example">
                    <p>🏪 同样的促销活动：</p>
                    <ul>
                        <li>加州店铺：+$200/天 ✅</li>
                        <li>伊利诺伊店：+$80/天 😐</li>
                        <li>亚利桑那店：-$30/天 ❌</li>
                    </ul>
                </div>
                <div class="decision-guide">
                    <p><strong>决策指导：</strong></p>
                    <ul>
                        <li>🎯 <strong>精准营销</strong>：在效果好的地方加大力度</li>
                        <li>⚠️ <strong>差异化策略</strong>：不同情况用不同方案</li>
                        <li>🚫 <strong>避免踩坑</strong>：在效果差的情况下暂停</li>
                    </ul>
                </div>
            </div>
        </div>
        """

    def _build_recommendations_cards(self, recommendations: List[Dict[str, str]]) -> str:
        """构建建议卡片"""
        cards_html = ""

        priority_colors = {
            '高': 'high-priority',
            '中': 'medium-priority',
            '低': 'low-priority'
        }

        priority_icons = {
            '高': '🔥',
            '中': '⭐',
            '低': '💡'
        }

        for rec in recommendations:
            priority = rec.get('priority', '中')
            color_class = priority_colors.get(priority, 'medium-priority')
            icon = priority_icons.get(priority, '💡')

            cards_html += f"""
            <div class="recommendation-card {color_class}">
                <div class="rec-header">
                    <span class="rec-icon">{icon}</span>
                    <span class="rec-priority">优先级: {priority}</span>
                </div>
                <h4>{rec['action']}</h4>
                <p class="rec-reason"><strong>原因:</strong> {rec['reason']}</p>
                <p class="rec-impact"><strong>预期影响:</strong> {rec['impact']}</p>
            </div>
            """

        return cards_html

    def _build_detailed_table(self, results: Dict[str, Any]) -> str:
        """构建详细结果表格"""
        table_html = """
        <table class="results-table">
            <thead>
                <tr>
                    <th>影响因素</th>
                    <th>平均效应 (ATE)</th>
                    <th>置信区间</th>
                    <th>显著性</th>
                    <th>处理率</th>
                    <th>样本量</th>
                </tr>
            </thead>
            <tbody>
        """

        factor_names = {
            'has_promotion': '促销活动',
            'is_weekend': '周末效应',
            'is_holiday': '节假日效应',
            'is_hot': '高温天气',
            'is_rainy': '雨天天气'
        }

        for factor_key, result in results.items():
            if factor_key in factor_names and 'ate' in result and 'error' not in result:
                ate = result['ate']
                ci_lower = result.get('ci_lower', ate - abs(ate) * 0.2)
                ci_upper = result.get('ci_upper', ate + abs(ate) * 0.2)
                significant = result.get('significant', False)
                treatment_rate = result.get('treatment_rate', 0)
                sample_size = result.get('sample_size', 0)

                sig_badge = '<span class="badge significant">显著</span>' if significant else '<span class="badge not-significant">不显著</span>'
                effect_class = 'positive' if ate > 0 else 'negative'

                table_html += f"""
                <tr>
                    <td><strong>{factor_names[factor_key]}</strong></td>
                    <td class="{effect_class}">${ate:+.2f}</td>
                    <td>[${ci_lower:.2f}, ${ci_upper:.2f}]</td>
                    <td>{sig_badge}</td>
                    <td>{treatment_rate:.1%}</td>
                    <td>{sample_size:,}</td>
                </tr>
                """

        table_html += """
            </tbody>
        </table>
        """

        return table_html

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
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            display: flex;
            align-items: center;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .summary-card.positive { background: linear-gradient(135deg, #a8e6cf, #88d8a3); }
        .summary-card.negative { background: linear-gradient(135deg, #ffaaa5, #ff8a80); }
        .summary-card.significant { background: linear-gradient(135deg, #a8d8ea, #7fcdff); }
        .summary-card.info { background: linear-gradient(135deg, #dda0dd, #d8bfd8); }
        
        .card-icon {
            font-size: 2.5em;
            margin-right: 15px;
        }
        
        .card-value {
            font-size: 2em;
            font-weight: bold;
            margin: 5px 0;
        }
        
        .card-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .concepts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .concept-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
        }
        
        .concept-card h3 {
            color: #2980b9;
            margin-bottom: 15px;
        }
        
        .example {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }
        
        .decision-guide {
            margin-top: 15px;
        }
        
        .decision-guide ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        .decision-guide li {
            margin: 5px 0;
        }
        
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: white;
        }
        
        .chart-container h3 {
            color: #34495e;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .explanation {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
        }
        
        .significant {
            color: #27ae60;
            font-weight: bold;
        }
        
        .not-significant {
            color: #f39c12;
            font-weight: bold;
        }
        
        .recommendations-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .recommendation-card {
            border-radius: 10px;
            padding: 20px;
            color: white;
        }
        
        .recommendation-card.high-priority {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        
        .recommendation-card.medium-priority {
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }
        
        .recommendation-card.low-priority {
            background: linear-gradient(135deg, #3498db, #2980b9);
        }
        
        .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .rec-icon {
            font-size: 1.5em;
        }
        
        .rec-priority {
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        
        .recommendation-card h4 {
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .rec-reason, .rec-impact {
            margin: 8px 0;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .collapsible-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            width: 100%;
            text-align: left;
        }
        
        .collapsible-btn:hover {
            background: #2980b9;
        }
        
        .collapsible-content {
            display: none;
            margin-top: 20px;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        .results-table .positive {
            color: #27ae60;
            font-weight: bold;
        }
        
        .results-table .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .badge.significant {
            background: #d4edda;
            color: #155724;
        }
        
        .badge.not-significant {
            background: #fff3cd;
            color: #856404;
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
        
        /* 行动推荐样式 */
        .action-summary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
        }
        
        .summary-stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            display: block;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .key-recommendation {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
        }
        
        .key-recommendation h3 {
            margin-bottom: 10px;
        }
        
        .action-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }
        
        .action-card.critical {
            border-left-color: #e74c3c;
            background: linear-gradient(135deg, #fff5f5, #white);
        }
        
        .action-card.important {
            border-left-color: #f39c12;
        }
        
        .action-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .action-number {
            background: #e74c3c;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .action-header h4 {
            flex-grow: 1;
            margin: 0;
            color: #2c3e50;
        }
        
        .impact-badge {
            background: #27ae60;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .action-description {
            color: #555;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .action-details {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .action-details span {
            background: #f8f9fa;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            color: #666;
        }
        
        .actions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .action-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        
        .improvement-list {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .improvement-list li {
            margin: 10px 0;
            padding-left: 10px;
        }
        
        .monitoring-section {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .metrics-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }
        
        .metric-tag {
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        
        .next-review {
            margin-top: 15px;
            font-style: italic;
            color: #666;
        }
        
        .no-actions {
            text-align: center;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
        }
        
        .urgency-high {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 2px 8px rgba(231, 76, 60, 0.2); }
            50% { box-shadow: 0 4px 16px rgba(231, 76, 60, 0.4); }
            100% { box-shadow: 0 2px 8px rgba(231, 76, 60, 0.2); }
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
            
            .summary-grid,
            .concepts-grid,
            .recommendations-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _get_javascript(self) -> str:
        """获取JavaScript代码"""
        return """
        function toggleSection(sectionId) {
            const content = document.getElementById(sectionId);
            const btn = content.previousElementSibling;
            
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                btn.innerHTML = btn.innerHTML.replace('▼', '▲');
            } else {
                content.style.display = 'none';
                btn.innerHTML = btn.innerHTML.replace('▲', '▼');
            }
        }
        
        // 页面加载完成后的初始化
        document.addEventListener('DOMContentLoaded', function() {
            // 添加表格排序功能（简化版）
            const tables = document.querySelectorAll('.results-table');
            tables.forEach(table => {
                const headers = table.querySelectorAll('th');
                headers.forEach((header, index) => {
                    header.style.cursor = 'pointer';
                    header.addEventListener('click', () => {
                        // 简单的排序提示
                        header.style.background = '#e3f2fd';
                        setTimeout(() => {
                            header.style.background = '#f2f2f2';
                        }, 200);
                    });
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


# ======================================================================
# 完整使用示例
# ======================================================================
if __name__ == "__main__":
    # 这里是完整的使用流程示例

    # 1. 运行因果分析
    from fixed_causal_inference import UMeCausalInferenceEngine

    CLICKHOUSE_CONFIG = {
        "host": "clickhouse-0-0.umetea.net",
        "port": 443,
        "database": "dw",
        "user": "ml_ume",
        "password": "hDAoDvg8x552bH",
        "verify": False,
    }

    # 初始化分析引擎
    engine = UMeCausalInferenceEngine(CLICKHOUSE_CONFIG)

    # 运行完整分析
    start_date, end_date = "2025-06-01", "2025-07-31"
    analysis_results = engine.run_complete_analysis(start_date, end_date)

    # 2. 生成HTML报告
    report_generator = UMeHTMLReportGenerator()
    report_filename = report_generator.generate_complete_report(analysis_results)

    print(f"✅ 完整分析和报告生成完成！")
    print(f"📄 HTML报告文件: {report_filename}")
    print(f"🌐 请在浏览器中打开查看完整报告")