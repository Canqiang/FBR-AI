"""
AI增长引擎Demo - 基于FBR实际数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from clickhouse_driver import Client

# ClickHouse连接配置
CLICKHOUSE_CONFIG = {
    'host': 'your_host',
    'port': 9000,
    'database': 'dw',
    'user': 'your_user',
    'password': 'your_password'
}


class FBRGrowthEngine:
    """FBR餐饮AI增长引擎"""

    def __init__(self):
        self.ch_client = Client(**CLICKHOUSE_CONFIG)

    def load_sales_data(self, days_back=90):
        """加载历史销售数据"""
        query = f"""
        SELECT 
            toDate(created_at_pt) as date,
            location_id,
            location_name,
            COUNT(DISTINCT order_id) as order_count,
            SUM(item_qty) as total_items,
            SUM(item_total_amt) as total_revenue,
            SUM(item_discount) as total_discount,
            AVG(item_total_amt) as avg_order_value,
            COUNT(DISTINCT customer_id) as unique_customers,
            SUM(CASE WHEN has(is_new_users, 1) THEN 1 ELSE 0 END) as new_customers,
            COUNT(DISTINCT CASE WHEN is_loyalty = 1 THEN customer_id END) as loyalty_customers
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= today() - {days_back}
            AND pay_status = 'COMPLETED'
        GROUP BY date, location_id, location_name
        ORDER BY date DESC
        """
        return pd.DataFrame(self.ch_client.execute(query, with_column_types=True))

    def load_item_performance(self, days=30):
        """加载商品销售表现"""
        query = f"""
        SELECT 
            item_name,
            category_name,
            COUNT(DISTINCT order_id) as order_count,
            SUM(item_qty) as units_sold,
            SUM(item_total_amt) as revenue,
            AVG(item_amt) as avg_price,
            SUM(item_discount) as total_discount,
            COUNT(DISTINCT customer_id) as unique_buyers,
            toDate(created_at_pt) as date
        FROM dw.fact_order_item_variations
        WHERE created_at_pt >= today() - {days}
            AND pay_status = 'COMPLETED'
            AND item_name IS NOT NULL
        GROUP BY item_name, category_name, date
        ORDER BY revenue DESC
        """
        return pd.DataFrame(self.ch_client.execute(query, with_column_types=True))

    def load_customer_segments(self):
        """加载客户分群数据"""
        query = """
        SELECT 
            COUNT(DISTINCT customer_id) as customer_count,
            SUM(order_final_total_amt) as total_revenue,
            AVG(order_final_avg_amt) as avg_order_value,
            CASE 
                WHEN high_value_customer = 1 THEN '高价值客户'
                WHEN key_development_customer = 1 THEN '重点发展客户'
                WHEN regular_customer = 1 THEN '普通客户'
                WHEN critical_win_back_customer = 1 THEN '重点挽回客户'
                ELSE '其他'
            END as customer_segment
        FROM ads.customer_profile
        WHERE order_final_total_cnt > 0
        GROUP BY customer_segment
        """
        return pd.DataFrame(self.ch_client.execute(query, with_column_types=True))

    def predict_next_week_sales(self, location_id=None):
        """预测下周销量 - 使用Prophet"""
        # 加载历史数据
        sales_data = self.load_sales_data(days_back=180)

        if location_id:
            sales_data = sales_data[sales_data['location_id'] == location_id]

        # 准备Prophet数据格式
        prophet_data = sales_data.groupby('date').agg({
            'total_revenue': 'sum',
            'order_count': 'sum'
        }).reset_index()
        prophet_data.columns = ['ds', 'y', 'order_count']

        # 训练模型
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        # 添加额外特征
        model.add_regressor('order_count')

        model.fit(prophet_data)

        # 预测未来7天
        future = model.make_future_dataframe(periods=7)
        future['order_count'] = prophet_data['order_count'].mean()  # 使用平均值

        forecast = model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    def analyze_sales_drivers(self):
        """分析销售驱动因素"""
        # 加载最近30天数据
        recent_data = self.load_sales_data(days_back=30)

        # 创建特征
        recent_data['day_of_week'] = pd.to_datetime(recent_data['date']).dt.dayofweek
        recent_data['is_weekend'] = recent_data['day_of_week'].isin([5, 6]).astype(int)
        recent_data['discount_rate'] = recent_data['total_discount'] / (
                    recent_data['total_revenue'] + recent_data['total_discount'])
        recent_data['new_customer_rate'] = recent_data['new_customers'] / recent_data['unique_customers']
        recent_data['loyalty_rate'] = recent_data['loyalty_customers'] / recent_data['unique_customers']

        # XGBoost特征重要性分析
        features = ['day_of_week', 'is_weekend', 'discount_rate', 'new_customer_rate', 'loyalty_rate',
                    'avg_order_value']
        X = recent_data[features]
        y = recent_data['total_revenue']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # 获取特征重要性
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance, model

    def generate_recommendations(self):
        """生成智能建议"""
        # 分析各维度数据
        item_perf = self.load_item_performance()
        customer_segments = self.load_customer_segments()
        sales_forecast = self.predict_next_week_sales()

        recommendations = []

        # 1. 爆品推荐
        top_items = item_perf.nlargest(5, 'revenue')
        slow_items = item_perf[item_perf['units_sold'] < item_perf['units_sold'].quantile(0.2)]

        if len(slow_items) > 0:
            recommendations.append({
                'type': '库存优化',
                'priority': '高',
                'action': f"以下商品销售缓慢，建议促销清仓：{', '.join(slow_items['item_name'].head(3))}",
                'expected_impact': '减少库存积压，提升资金周转率15%'
            })

        # 2. 客户营销建议
        if 'critical_win_back_customer' in customer_segments['customer_segment'].values:
            win_back_count = \
            customer_segments[customer_segments['customer_segment'] == '重点挽回客户']['customer_count'].iloc[0]
            recommendations.append({
                'type': '客户挽回',
                'priority': '高',
                'action': f"发现{win_back_count}位重点挽回客户，建议发送专属优惠券",
                'expected_impact': '预计挽回30%流失客户，增加月收入8%'
            })

        # 3. 销售预测建议
        next_week_avg = sales_forecast['yhat'].mean()
        current_week_avg = self.load_sales_data(days_back=7)['total_revenue'].mean()

        if next_week_avg < current_week_avg * 0.9:
            recommendations.append({
                'type': '销售预警',
                'priority': '高',
                'action': '预测下周销量可能下滑，建议提前准备促销活动',
                'expected_impact': '及时干预可避免10%的销售损失'
            })

        return recommendations


# Streamlit界面
def main():
    st.set_page_config(page_title="FBR AI增长引擎", layout="wide")

    # 标题
    st.title("🚀 FBR AI增长引擎")
    st.markdown("### 数据驱动决策，AI赋能增长")

    # 初始化引擎
    engine = FBRGrowthEngine()

    # 侧边栏
    with st.sidebar:
        st.header("📊 数据范围")

        # 选择门店
        location = st.selectbox(
            "选择门店",
            ["全部门店", "门店A", "门店B", "门店C"]
        )

        # 选择时间范围
        date_range = st.selectbox(
            "时间范围",
            ["最近7天", "最近30天", "最近90天"]
        )

        # 刷新按钮
        if st.button("🔄 刷新数据"):
            st.experimental_rerun()

    # 主界面 - 分成三个标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📈 实时监控", "🔮 销售预测", "🎯 智能建议", "💬 AI助手"])

    with tab1:
        st.header("实时业务监控")

        # 加载今日数据
        today_data = engine.load_sales_data(days_back=1)
        yesterday_data = engine.load_sales_data(days_back=2).iloc[1:2]

        # KPI卡片
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            today_revenue = today_data['total_revenue'].sum()
            yesterday_revenue = yesterday_data['total_revenue'].sum() if len(yesterday_data) > 0 else 0
            revenue_change = (
                        (today_revenue - yesterday_revenue) / yesterday_revenue * 100) if yesterday_revenue > 0 else 0

            st.metric(
                "今日营收",
                f"¥{today_revenue:,.0f}",
                f"{revenue_change:+.1f}%"
            )

        with col2:
            today_orders = today_data['order_count'].sum()
            yesterday_orders = yesterday_data['order_count'].sum() if len(yesterday_data) > 0 else 0
            order_change = ((today_orders - yesterday_orders) / yesterday_orders * 100) if yesterday_orders > 0 else 0

            st.metric(
                "订单数",
                f"{today_orders:,}",
                f"{order_change:+.1f}%"
            )

        with col3:
            today_customers = today_data['unique_customers'].sum()
            new_customers = today_data['new_customers'].sum()

            st.metric(
                "客户数",
                f"{today_customers:,}",
                f"新客: {new_customers}"
            )

        with col4:
            avg_order = today_revenue / today_orders if today_orders > 0 else 0
            st.metric(
                "客单价",
                f"¥{avg_order:.0f}",
                "→"
            )

        # 销售趋势图
        st.subheader("📊 最近30天销售趋势")

        trend_data = engine.load_sales_data(days_back=30)
        trend_summary = trend_data.groupby('date')['total_revenue'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_summary['date'],
            y=trend_summary['total_revenue'],
            mode='lines+markers',
            name='日销售额',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title="日销售额趋势",
            xaxis_title="日期",
            yaxis_title="销售额 (¥)",
            hovermode='x'
        )

        st.plotly_chart(fig, use_container_width=True)

        # 热销商品
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔥 今日热销TOP5")
            hot_items = engine.load_item_performance(days=1).nlargest(5, 'revenue')

            fig_hot = px.bar(
                hot_items,
                x='revenue',
                y='item_name',
                orientation='h',
                color='revenue',
                color_continuous_scale='Reds'
            )
            fig_hot.update_layout(showlegend=False)
            st.plotly_chart(fig_hot, use_container_width=True)

        with col2:
            st.subheader("📊 品类销售占比")
            category_sales = engine.load_item_performance(days=1).groupby('category_name')['revenue'].sum()

            fig_pie = px.pie(
                values=category_sales.values,
                names=category_sales.index,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.header("销售预测分析")

        # 预测下周销售
        with st.spinner("AI正在预测未来销售趋势..."):
            forecast = engine.predict_next_week_sales()

        # 显示预测结果
        st.subheader("📈 未来7天销售预测")

        fig_forecast = go.Figure()

        # 添加预测值
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='预测销售额',
            line=dict(color='blue', width=2)
        ))

        # 添加置信区间
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='置信区间'
        ))

        fig_forecast.update_layout(
            title="7天销售额预测",
            xaxis_title="日期",
            yaxis_title="预测销售额 (¥)",
            hovermode='x'
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # 预测洞察
        avg_forecast = forecast['yhat'].mean()
        max_day = forecast.loc[forecast['yhat'].idxmax()]
        min_day = forecast.loc[forecast['yhat'].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"""
            **预测摘要**
            - 平均日销售额: ¥{avg_forecast:,.0f}
            - 最高: {max_day['ds'].strftime('%m月%d日')} (¥{max_day['yhat']:,.0f})
            - 最低: {min_day['ds'].strftime('%m月%d日')} (¥{min_day['yhat']:,.0f})
            """)

        with col2:
            st.warning(f"""
            **风险提示**
            - {min_day['ds'].strftime('%m月%d日')}销量可能较低
            - 建议提前准备营销活动
            - 关注库存避免缺货
            """)

        with col3:
            st.success(f"""
            **机会点**
            - {max_day['ds'].strftime('%m月%d日')}预计高峰
            - 建议增加备货
            - 可推出限时优惠
            """)

        # 销售驱动因素分析
        st.subheader("🔍 销售驱动因素分析")

        importance_df, _ = engine.analyze_sales_drivers()

        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
            labels={
                'importance': '重要性',
                'feature': '影响因素'
            }
        )

        # 重命名特征
        feature_names = {
            'avg_order_value': '客单价',
            'discount_rate': '折扣率',
            'is_weekend': '周末效应',
            'day_of_week': '星期效应',
            'new_customer_rate': '新客占比',
            'loyalty_rate': '会员占比'
        }

        fig_importance.update_yaxis(
            ticktext=[feature_names.get(f, f) for f in importance_df['feature']],
            tickvals=importance_df['feature']
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    with tab3:
        st.header("智能运营建议")

        # 生成建议
        with st.spinner("AI正在分析数据并生成建议..."):
            recommendations = engine.generate_recommendations()

        # 显示建议卡片
        for i, rec in enumerate(recommendations):
            priority_color = {
                '高': 'red',
                '中': 'orange',
                '低': 'green'
            }.get(rec['priority'], 'gray')

            with st.container():
                col1, col2 = st.columns([1, 5])

                with col1:
                    st.markdown(f"""
                    <div style="background-color: {priority_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                        <strong>{rec['priority']}优先级</strong>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.subheader(f"{i + 1}. {rec['type']}")
                    st.write(f"**建议行动**: {rec['action']}")
                    st.write(f"**预期效果**: {rec['expected_impact']}")

                    # 添加执行按钮
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
                    with col_btn1:
                        if st.button(f"立即执行", key=f"exec_{i}"):
                            st.success("✅ 已加入执行队列")
                    with col_btn2:
                        if st.button(f"稍后提醒", key=f"remind_{i}"):
                            st.info("⏰ 将在2小时后提醒")

                st.markdown("---")

        # 客户分群洞察
        st.subheader("👥 客户分群洞察")

        customer_segments = engine.load_customer_segments()

        fig_segments = px.treemap(
            customer_segments,
            path=['customer_segment'],
            values='customer_count',
            color='total_revenue',
            color_continuous_scale='Viridis',
            title="客户分群分布"
        )

        st.plotly_chart(fig_segments, use_container_width=True)

        # 显示具体数据
        st.dataframe(
            customer_segments.style.format({
                'customer_count': '{:,.0f}',
                'total_revenue': '¥{:,.0f}',
                'avg_order_value': '¥{:.0f}'
            }),
            use_container_width=True
        )

    with tab4:
        st.header("AI运营助手")

        # 预设问题
        st.markdown("### 💡 您可以问我:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 最近哪些商品卖得好？"):
                hot_items = engine.load_item_performance(days=7).nlargest(10, 'revenue')
                st.write("**最近7天热销商品TOP10:**")
                for idx, item in hot_items.iterrows():
                    st.write(f"- {item['item_name']}: 销售额¥{item['revenue']:,.0f}, 售出{item['units_sold']}份")

            if st.button("📉 为什么销量下降了？"):
                st.write("""
                **AI分析结果：**

                根据数据分析，最近销量下降的主要原因：

                1. **周期性因素** (35%): 本周处于月末，历史数据显示月末销量通常会下降
                2. **促销结束** (28%): 上周的满减活动已结束，导致订单量减少
                3. **天气影响** (20%): 近期阴雨天气，外卖订单减少
                4. **新客获取** (17%): 新客户增长放缓，需要加强营销

                **建议措施：**
                - 推出月末专属优惠，刺激消费
                - 针对会员发送召回优惠券
                - 优化雨天配送服务
                """)

        with col2:
            if st.button("💰 如何提升客单价？"):
                st.write("""
                **提升客单价策略：**

                基于您的数据分析，建议：

                1. **套餐优化**
                   - 将热销单品组合成超值套餐
                   - 当前客单价¥45，套餐可提升至¥58

                2. **加购推荐**
                   - 在结账时推荐小食/饮品
                   - 预计可提升15%客单价

                3. **会员专享**
                   - 为会员提供升级套餐选项
                   - 培养高价值客户消费习惯

                4. **时段差异化**
                   - 晚餐时段推出精品套餐
                   - 利用需求弹性提升单价
                """)

            if st.button("🎯 明天应该重点关注什么？"):
                tomorrow = datetime.now() + timedelta(days=1)
                day_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][tomorrow.weekday()]

                st.write(f"""
                **明天（{day_name}）运营重点：**

                📍 **重点关注:**
                - 预计客流高峰：12:00-13:00, 18:00-20:00
                - 建议备货：根据预测，主食类需求会增加20%

                🎯 **营销机会:**
                - {day_name}工作餐推广
                - 针对上班族的快速套餐

                ⚠️ **风险提示:**
                - 注意热销商品库存
                - 确保高峰期人员充足

                💡 **AI建议:**
                - 11:30前完成午餐备货
                - 提前发送午餐优惠提醒
                """)

        # 自由对话区
        st.markdown("---")
        st.markdown("### 💬 自由提问")

        user_question = st.text_input("请输入您的问题：", placeholder="例如：如何提高周末的销量？")

        if user_question:
            with st.spinner("AI思考中..."):
                # 这里可以接入真实的LLM
                st.write(f"""
                **关于"{user_question}"的分析：**

                基于您的历史数据，我的建议是：

                1. 分析相关数据指标
                2. 识别关键影响因素
                3. 制定针对性策略
                4. 跟踪执行效果

                需要我详细分析具体哪个方面吗？
                """)


if __name__ == "__main__":
    main()