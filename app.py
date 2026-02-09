#!/usr/bin/env python
# coding: utf-8
"""
Spotify 数据分析仪表板
Streamlit 网页应用，提供交互式音乐数据分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置页面配置
st.set_page_config(
    page_title="Spotify 数据分析",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 自定义 CSS 样式
st.markdown("""
<style>
    .main {
        background-color: #1DB954;
    }
    .stApp {
        background-color: #191414;
    }
    .css-1d391kg {
        background-color: #191414;
    }
    .css-18ni7ap {
        background-color: #191414;
    }
    h1, h2, h3 {
        color: #1DB954;
    }
    .stSelectbox > div > div > div {
        background-color: #191414;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 数据加载和预处理模块
# ============================================================================

@st.cache_data
def load_and_process_data():
    """加载并预处理数据"""
    # 加载数据
    sp_tracks = pd.read_csv('data/tracks.csv')
    sp_feature = pd.read_csv('data/SpotifyFeatures.csv')

    # 处理音轨数据
    sp_tracks['release_date'] = pd.to_datetime(sp_tracks['release_date'])
    sp_tracks['year'] = sp_tracks['release_date'].dt.year
    sp_tracks['duration'] = sp_tracks['duration_ms'].apply(lambda x: round(x / 1000))
    sp_tracks.drop('duration_ms', inplace=True, axis=1)

    # 处理特征数据
    sp_feature['duration'] = sp_feature['duration_ms'].apply(lambda x: round(x / 1000))

    return sp_tracks, sp_feature

def filter_by_year(df, year_range):
    """根据年份范围过滤数据"""
    if year_range[0] == year_range[1]:
        return df[df['year'] == year_range[0]]
    return df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

def filter_by_genre(df, genres):
    """根据音乐类型过滤数据"""
    if '全部类型' in genres or len(genres) == 0:
        return df
    return df[df['genre'].isin(genres)]

def get_yearly_stats(df):
    """获取年度统计信息"""
    yearly = df.groupby('year').agg({
        'popularity': ['mean', 'count'],
        'duration': 'mean'
    }).round(2)
    yearly.columns = ['平均流行度', '歌曲数量', '平均时长']
    return yearly.reset_index()

# ============================================================================
# 可视化函数
# ============================================================================

def plot_popularity_analysis(df):
    """绘制流行度分析图表"""
    st.subheader("🎯 流行度分析")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 最受欢迎的 10 首歌曲")
        top_songs = df.nlargest(10, 'popularity')[['name', 'artists', 'popularity', 'year']]
        fig = px.bar(
            top_songs,
            x='popularity',
            y='name',
            color='popularity',
            color_continuous_scale='Greens',
            orientation='h',
            title="",
            text='popularity'
        )
        fig.update_layout(
            yaxis_title="歌曲名称",
            xaxis_title="流行度",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(textposition='inside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 流行度分布")
        fig = px.histogram(
            df,
            x='popularity',
            nbins=30,
            color_discrete_sequence=['#1DB954'],
            title=""
        )
        fig.update_layout(
            xaxis_title="流行度",
            yaxis_title="歌曲数量",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_correlation_analysis(df):
    """绘制相关性分析图表"""
    st.subheader("📊 相关性分析")

    # 选择数值型特征
    numeric_features = df[['popularity', 'danceability', 'energy', 'loudness',
                            'acousticness', 'valence', 'tempo', 'duration']]

    # 计算相关性矩阵
    corr_matrix = numeric_features.corr(method='pearson')

    # 相关性热力图
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="相关系数")
    ))

    fig.update_layout(
        title="特征相关性热力图",
        height=500,
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # 散点图
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 能量 vs 响度")
        fig = px.scatter(
            df.sample(min(500, len(df))),
            x='energy',
            y='loudness',
            color='popularity',
            color_continuous_scale='Greens',
            opacity=0.7,
            title=""
        )
        fig.update_layout(
            xaxis_title="能量",
            yaxis_title="响度",
            height=350,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_trace(go.Scatter(
            x=df['energy'],
            y=df['loudness'],
            mode='lines',
            line=dict(color='white', dash='dash', width=2),
            name='趋势线',
            showlegend=False
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 流行度 vs 声学特征")
        fig = px.scatter(
            df.sample(min(500, len(df))),
            x='acousticness',
            y='popularity',
            color='energy',
            color_continuous_scale='Viridis',
            opacity=0.7,
            title=""
        )
        fig.update_layout(
            xaxis_title="声学特征",
            yaxis_title="流行度",
            height=350,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_time_series_analysis(df):
    """绘制时间序列分析图表"""
    st.subheader("📅 时间序列分析")

    yearly_stats = get_yearly_stats(df)

    # 年份分布
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 各年份歌曲数量分布")
        fig = px.bar(
            yearly_stats,
            x='year',
            y='歌曲数量',
            color='歌曲数量',
            color_continuous_scale='Greens',
            title=""
        )
        fig.update_layout(
            xaxis_title="年份",
            yaxis_title="歌曲数量",
            height=350,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 平均时长随年份变化")
        fig = px.line(
            yearly_stats,
            x='year',
            y='平均时长',
            title=""
        )
        fig.update_traces(line_color='#1DB954', line_width=3)
        fig.update_layout(
            xaxis_title="年份",
            yaxis_title="平均时长 (秒)",
            height=350,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_scatter(
            x=yearly_stats['year'],
            y=yearly_stats['平均时长'],
            mode='markers',
            marker=dict(size=8, color='#1DB954'),
            name='',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # 流行度随时间变化
    st.markdown("### 流行度随时间变化")
    fig = px.line(
        yearly_stats,
        x='year',
        y='平均流行度',
        title=""
    )
    fig.update_traces(line_color='#1DB954', line_width=3)
    fig.update_layout(
        xaxis_title="年份",
        yaxis_title="平均流行度",
        height=300,
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.add_scatter(
        x=yearly_stats['year'],
        y=yearly_stats['平均流行度'],
        mode='markers',
        marker=dict(size=8, color='#1DB954'),
        name='',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_genre_analysis(sp_feature):
    """绘制音乐类型分析图表"""
    st.subheader("🎼 音乐类型分析")

    # 各类型平均时长
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 各类型平均时长对比")
        genre_duration = sp_feature.groupby('genre')['duration'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=genre_duration.values,
            y=genre_duration.index,
            orientation='h',
            color=genre_duration.values,
            color_continuous_scale='Greens',
            title=""
        )
        fig.update_layout(
            xaxis_title="平均时长 (秒)",
            yaxis_title="音乐类型",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 各类型流行度排名")
        genre_popularity = sp_feature.groupby('genre')['danceability'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=genre_popularity.values,
            y=genre_popularity.index,
            orientation='h',
            color=genre_popularity.values,
            color_continuous_scale='Viridis',
            title=""
        )
        fig.update_layout(
            xaxis_title="平均可舞性",
            yaxis_title="音乐类型",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 类型分布
    st.markdown("### 音乐类型分布")
    genre_counts = sp_feature['genre'].value_counts()
    fig = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Greens_r
    )
    fig.update_layout(
        height=400,
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_overview_stats(df, sp_feature):
    """显示概览统计"""
    st.markdown("## 📈 数据概览")

    total_tracks = len(df)
    avg_popularity = df['popularity'].mean()
    avg_duration = df['duration'].mean()
    years_range = f"{df['year'].min()} - {df['year'].max()}"
    total_genres = len(sp_feature['genre'].unique())

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="总歌曲数",
            value=f"{total_tracks:,}",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="平均流行度",
            value=f"{avg_popularity:.2f}",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="平均时长",
            value=f"{avg_duration:.0f}秒",
            delta_color="normal"
        )

    with col4:
        st.metric(
            label="年份范围",
            value=years_range,
            delta_color="normal"
        )

    with col5:
        st.metric(
            label="音乐类型",
            value=f"{total_genres}",
            delta_color="normal"
        )

# ============================================================================
# 主应用
# ============================================================================

def main():
    """主函数"""
    # 加载数据
    sp_tracks, sp_feature = load_and_process_data()

    # 侧边栏
    st.sidebar.markdown("# 🎵 Spotify 数据分析")
    st.sidebar.markdown("---")

    # 年份范围选择
    min_year = int(sp_tracks['year'].min())
    max_year = int(sp_tracks['year'].max())

    year_range = st.sidebar.slider(
        "选择年份范围",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    # 音乐类型选择
    genres = ['全部类型'] + sorted(sp_feature['genre'].unique().tolist())
    selected_genres = st.sidebar.multiselect(
        "选择音乐类型",
        genres,
        default=['全部类型']
    )

    # 分析选项
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 分析视图")
    analysis_options = st.sidebar.multiselect(
        "选择要显示的分析",
        ["数据概览", "流行度分析", "相关性分析", "时间序列分析", "音乐类型分析"],
        default=["数据概览", "流行度分析", "相关性分析", "时间序列分析", "音乐类型分析"]
    )

    # 过滤数据
    filtered_tracks = filter_by_year(sp_tracks, year_range)
    filtered_features = filter_by_genre(sp_feature, selected_genres)

    # 主内容区
    st.title("🎵 Spotify 音乐数据分析仪表板")
    st.markdown(f"**数据范围:** {year_range[0]} - {year_range[1]} 年")
    if '全部类型' not in selected_genres and len(selected_genres) > 0:
        st.markdown(f"**音乐类型:** {', '.join(selected_genres)}")
    st.markdown("---")

    # 显示选中的分析视图
    if "数据概览" in analysis_options:
        show_overview_stats(filtered_tracks, filtered_features)
        st.markdown("---")

    if "流行度分析" in analysis_options:
        plot_popularity_analysis(filtered_tracks)
        st.markdown("---")

    if "相关性分析" in analysis_options:
        plot_correlation_analysis(filtered_tracks)
        st.markdown("---")

    if "时间序列分析" in analysis_options:
        plot_time_series_analysis(filtered_tracks)
        st.markdown("---")

    if "音乐类型分析" in analysis_options:
        plot_genre_analysis(filtered_features)

    # 页脚
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"Spotify 数据分析应用 | 基于 {len(filtered_tracks)} 首歌曲"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
