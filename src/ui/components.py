import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Callable


def MetricsCard(title: str, value: str, delta: str = None):
    """指标卡片组件"""
    container = st.container()
    with container:
        st.metric(
            label=title,
            value=value,
            delta=delta
        )


def SalesChart(data: pd.DataFrame):
    """销售趋势图表"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['revenue'],
        mode='lines+markers',
        name='营收',
        line=dict(color='#1f77b4', width=2)
    ))

    # 添加订单数（次坐标轴）
    if 'orders' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['orders'],
            mode='lines+markers',
            name='订单数',
            line=dict(color='#ff7f0e', width=2),
            yaxis='y2'
        ))

    fig.update_layout(
        title="销售趋势",
        xaxis_title="日期",
        yaxis_title="营收 (¥)",
        yaxis2=dict(
            title="订单数",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def CustomerSegmentChart(segments_data: Dict[str, Any]):
    """客户分群图表"""
    segments = segments_data.get('segments', [])

    if not segments:
        st.warning("暂无客户分群数据")
        return

    # 准备数据
    df = pd.DataFrame(segments)

    # 创建子图
    col1, col2 = st.columns(2)

    with col1:
        # 客户数量分布
        fig_pie = px.pie(
            df,
            values='customer_count',
            names='segment_name',
            title='客户数量分布'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # 营收贡献
        fig_bar = px.bar(
            df,
            x='segment_name',
            y='total_revenue',
            title='各分群营收贡献',
            color='avg_order_value',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def ItemPerformanceTable(items: List[Dict]):
    """商品表现表格"""
    if not items:
        st.warning("暂无商品数据")
        return

    # 转换为DataFrame
    df = pd.DataFrame(items)

    # 确保必要的列存在
    required_columns = ['item_name', 'revenue', 'units_sold']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"数据缺少必要的列: {col}")
            return

    # 确保 revenue 是数字类型
    try:
        # 转换为数字，无法转换的设为 0
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"转换营收数据时出错: {e}")
        df['revenue'] = 0

    # 格式化显示
    df['revenue_fmt'] = df['revenue'].apply(lambda x: f"¥{float(x):,.0f}" if pd.notna(x) else "¥0")

    # 处理 performance_score（如果存在）
    if 'performance_score' in df.columns:
        try:
            df['performance_score'] = pd.to_numeric(df['performance_score'], errors='coerce').fillna(0)
            df['performance_bar'] = df['performance_score'].apply(
                lambda x: '🟩' * max(0, min(5, int(float(x) * 5)))
            )
        except:
            df['performance_bar'] = '🟩🟩🟩'  # 默认3个方块
    else:
        # 如果没有 performance_score，根据 revenue 生成
        max_revenue = df['revenue'].max() if df['revenue'].max() > 0 else 1
        df['performance_bar'] = df['revenue'].apply(
            lambda x: '🟩' * max(1, min(5, int((x / max_revenue) * 5)))
        )

    # 处理 category 列（如果不存在则设为默认值）
    if 'category' not in df.columns:
        df['category'] = '未分类'

    # 确保 units_sold 是数字
    if 'units_sold' in df.columns:
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0).astype(int)

    # 显示表格
    display_columns = ['item_name', 'category', 'revenue_fmt', 'units_sold', 'performance_bar']
    display_df = df[display_columns].copy()
    display_df.columns = ['商品名称', '类别', '销售额', '售出数量', '表现']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def RecommendationCard(recommendation: Dict[str, Any]):
    """建议卡片"""
    priority_colors = {
        '高': 'red',
        '中': 'orange',
        '低': 'green'
    }

    priority = recommendation.get('priority', '中')
    color = priority_colors.get(priority, 'gray')

    with st.container():
        col1, col2 = st.columns([1, 5])

        with col1:
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>{priority}优先级</strong>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader(recommendation.get('action', ''))
            st.write(f"**原因**: {recommendation.get('reason', '')}")
            st.write(f"**预期效果**: {recommendation.get('expected_impact', '')}")

            if st.button("执行", key=f"exec_{recommendation.get('type', '')}_{id(recommendation)}"):
                st.success("已加入执行队列")


def ChatInterface(send_callback: Callable):
    """聊天界面组件"""
    # 显示聊天历史
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message['role']):
            st.write(message['content'])
            st.caption(message['timestamp'].strftime('%H:%M'))

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)

        # 调用回调处理消息
        send_callback(prompt)

        # 重新运行以显示回复
        st.rerun()

