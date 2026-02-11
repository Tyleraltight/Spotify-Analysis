#!/usr/bin/env python
# coding: utf-8
"""
Spotify 数据分析仪表板
Streamlit 网页应用，提供交互式音乐数据分析功能
Redesigned: SaaS-style professional dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Spotify Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# matplotlib CJK font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Design Tokens
# ============================================================================
FONT_STACK = "'Inter','Segoe UI',Roboto,Helvetica,Arial,sans-serif"
BG_PAGE = "#F5F7FA"
BG_CARD = "#FFFFFF"
BG_SIDEBAR = "#1E293B"
TEXT_PRIMARY = "#2D3748"
TEXT_SECONDARY = "#718096"
TEXT_SIDEBAR = "#F1F5F9"
BORDER_COLOR = "#E2E8F0"
ACCENT = "#4F46E5"
HEADER_BG = "#F1F5F9"

# Cluster colour palette (cold-tone, professional)
CLUSTER_COLORS = ["#4F46E5", "#0EA5E9", "#64748B", "#14B8A6", "#6366F1",
                  "#8B5CF6", "#06B6D4", "#475569", "#0D9488", "#7C3AED"]

# ============================================================================
# Global CSS Injection
# ============================================================================
st.markdown(f"""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global font (scoped to avoid breaking Streamlit dataframe column menus) ── */
    .stApp, .stApp *, .stApp *::before, .stApp *::after {{
        font-family: {FONT_STACK} !important;
    }}
    /* Restore default font for Glide Data Grid popups so sort/filter menus render correctly */
    div[id^="portal"], div[id^="portal"] *, div[id^="portal"] *::before, div[id^="portal"] *::after,
    .gdg-style, .gdg-style *, .click-outside-ignore, .click-outside-ignore * {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif !important;
        font-size: revert !important;
    }}

    /* ── Page background ── */
    .stApp {{
        background-color: {BG_PAGE} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Sidebar (light, harmonised) ── */
    section[data-testid="stSidebar"] {{
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {{
        color: #475569 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #334155 !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: #E2E8F0 !important;
        margin: 16px 0 !important;
    }}
    /* Relaxed spacing between sidebar controls */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        padding-bottom: 8px;
    }}

    /* Sidebar multiselect dropdown */
    section[data-testid="stSidebar"] .stMultiSelect > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
    }}
    /* Multiselect tags — soft indigo fusion */
    section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"],
    .stMultiSelect span[data-baseweb="tag"] {{
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 6px !important;
    }}
    section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] span,
    .stMultiSelect span[data-baseweb="tag"] span {{
        color: #4F46E5 !important;
    }}
    /* Tag close icon */
    section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] svg,
    .stMultiSelect span[data-baseweb="tag"] svg {{
        fill: #4F46E5 !important;
    }}

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }}

    /* ── Body text ── */
    p, span, label, .stMarkdown {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── Section spacing ── */
    .section-spacer {{
        margin-bottom: 30px;
    }}

    /* ── Card system ── */
    .saas-card {{
        background-color: {BG_CARD};
        border-radius: 16px;
        padding: 25px;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }}
    .saas-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05),
                    0 4px 6px -2px rgba(0,0,0,0.03);
    }}

    /* ── Metric card ── */
    .metric-card {{
        background-color: {BG_CARD};
        border-radius: 16px;
        padding: 24px 25px;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03);
        transition: all 0.3s ease;
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05),
                    0 4px 6px -2px rgba(0,0,0,0.03);
    }}
    .metric-icon {{
        font-size: 2.2rem;
        line-height: 1;
        flex-shrink: 0;
        width: 52px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(79,70,229,0.04));
    }}
    .metric-body {{
        flex: 1;
        min-width: 0;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        line-height: 1.15;
        margin: 0 0 4px 0;
    }}
    .metric-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {TEXT_SECONDARY};
        margin: 4px 0 0 0;
    }}

    /* ── Page header ── */
    .page-header {{
        padding: 10px 0 6px 0;
        margin-bottom: 28px;
    }}
    .page-header h1 {{
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: {TEXT_PRIMARY} !important;
        margin: 0 0 6px 0 !important;
    }}
    .page-header .subtitle {{
        font-size: 0.95rem;
        color: {TEXT_SECONDARY};
        margin: 0;
    }}

    /* ── Section titles ── */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin: 0 0 5px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* ── Buttons (Ghost style) ── */
    .stButton > button {{
        background-color: transparent !important;
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: all 0.25s ease !important;
    }}
    .stButton > button:hover {{
        background-color: #F1F5F9 !important;
        border-color: #4F46E5 !important;
        color: #4F46E5 !important;
        transform: translateY(-1px);
        box-shadow: none !important;
    }}
    .stButton > button:active,
    .stButton > button:focus {{
        background-color: #EEF2FF !important;
        border-color: #4F46E5 !important;
        color: #4F46E5 !important;
        box-shadow: none !important;
    }}

    /* ── Dataframe styling ── */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden;
    }}
    [data-testid="stDataFrame"] table th {{
        background-color: {HEADER_BG} !important;
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stDataFrame"] table tr:nth-child(even) {{
        background-color: #F8FAFC !important;
    }}

    /* ── Plotly charts: remove default streamlit container padding ── */
    [data-testid="stPlotlyChart"] > div {{
        border-radius: 12px !important;
    }}

    /* ── Slider (clean track + indigo handle) ── */
    .stSlider > div > div > div > div {{
        background-color: {ACCENT} !important;
    }}
    /* Slider track (unfilled portion) */
    .stSlider [data-testid="stThumbValue"] {{
        color: #4F46E5 !important;
        font-weight: 600 !important;
    }}
    .stSlider > div > div {{
        color: #475569 !important;
    }}
    section[data-testid="stSidebar"] .stSlider > div > div > div {{
        background-color: #E2E8F0 !important;
    }}
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
        background-color: #4F46E5 !important;
    }}

    /* ── Hide default streamlit branding elements ── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Divider / separator ── */
    .soft-divider {{
        border: none;
        border-top: 1px solid {BORDER_COLOR};
        margin: 30px 0;
    }}

    /* ── Footer ── */
    .app-footer {{
        text-align: center;
        color: {TEXT_SECONDARY};
        font-size: 0.82rem;
        padding: 20px 0 10px 0;
        border-top: 1px solid {BORDER_COLOR};
        margin-top: 40px;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Plotly Layout Helpers
# ============================================================================
def _base_layout(**overrides):
    """Return a dict of common Plotly layout settings."""
    base = dict(
        template="plotly_white",
        font=dict(family=FONT_STACK.replace("'", ""), color=TEXT_PRIMARY, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(font=dict(family=FONT_STACK.replace("'", ""), size=11)),
    )
    base.update(overrides)
    return base


def _cluster_color_map(n):
    """Build {0: color, 1: color, ...} for n clusters."""
    return {str(i): CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n)}


# ============================================================================
# Data Loading & Processing (preserved)
# ============================================================================
@st.cache_data
def load_and_process_data():
    """加载并预处理数据"""
    sp_tracks = pd.read_csv('data/tracks.csv')
    sp_feature = pd.read_csv('data/SpotifyFeatures.csv')

    sp_tracks['release_date'] = pd.to_datetime(sp_tracks['release_date'])
    sp_tracks['year'] = sp_tracks['release_date'].dt.year
    sp_tracks['duration'] = sp_tracks['duration_ms'].apply(lambda x: round(x / 1000))
    sp_tracks.drop('duration_ms', inplace=True, axis=1)

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
# K-Means Clustering (core logic preserved & wired up)
# ============================================================================
@st.cache_data
def run_kmeans(df, n_clusters, feature_cols):
    """Run K-Means on selected numeric features and return labels."""
    valid_cols = [c for c in feature_cols if c in df.columns]
    if len(valid_cols) < 2 or len(df) < n_clusters:
        return None, None
    data = df[valid_cols].dropna()
    if len(data) < n_clusters:
        return None, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return labels, data.index


# ============================================================================
# Card Wrappers
# ============================================================================
def card_open():
    return '<div class="saas-card">'

def card_close():
    return '</div>'

def render_in_card(title_icon, title_text, content_fn):
    """Render streamlit content inside a card using columns trick."""
    st.markdown(f'<div class="section-title">{title_icon} {title_text}</div>', unsafe_allow_html=True)
    st.markdown(card_open(), unsafe_allow_html=True)
    content_fn()
    st.markdown(card_close(), unsafe_allow_html=True)


# ============================================================================
# Metric Cards (custom HTML)
# ============================================================================
def render_metric_card(icon, value, label, subtitle=""):
    sub_html = f'<p class="metric-sub">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-body">
            <p class="metric-value">{value}</p>
            <p class="metric-label">{label}</p>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_metrics(df, sp_feature):
    """Top row: 3 metric cards."""
    total_tracks = int(len(df))
    avg_popularity = float(df['popularity'].mean()) if len(df) else 0.0
    avg_duration = float(df['duration'].mean()) if len(df) else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("🎵", f"{total_tracks:,}", "Total Tracks", "Songs in current filter range")
    with c2:
        render_metric_card("🔥", f"{avg_popularity:.1f}", "Avg Popularity", "Mean popularity score (0–100)")
    with c3:
        render_metric_card("⏱️", f"{avg_duration:.0f}s", "Avg Duration", "Mean track length in seconds")


# ============================================================================
# Visualisation Functions
# ============================================================================
def plot_popularity_analysis(df):
    """Popularity section — scatter + distribution."""
    st.markdown('<div class="section-title">🎯 Popularity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Dashboard: scatter plot + distribution side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Danceability vs Energy**")
        if len(df) > 0:
            sample_df = df.sample(min(1200, len(df)))
            x_col = 'danceability' if 'danceability' in df.columns else 'duration'
            y_col = 'energy' if 'energy' in df.columns else 'popularity'
            fig = px.scatter(
                sample_df, x=x_col, y=y_col,
                color='popularity' if 'popularity' in df.columns else None,
                color_continuous_scale=[[0, "#E0E7FF"], [0.5, "#818CF8"], [1, "#4F46E5"]],
                opacity=0.7, title=""
            )
            fig.update_layout(**_base_layout(
                height=400,
                xaxis_title="Danceability" if x_col == 'danceability' else "Duration (s)",
                yaxis_title="Energy" if y_col == 'energy' else "Popularity",
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for the current filter.")
        st.markdown(card_close(), unsafe_allow_html=True)

    with col2:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Popularity Distribution**")
        fig = px.histogram(
            df, x='popularity', nbins=30,
            color_discrete_sequence=[ACCENT], title=""
        )
        fig.update_layout(**_base_layout(
            height=400,
            xaxis_title="Popularity", yaxis_title="Track Count",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Top 10 songs
    st.markdown(card_open(), unsafe_allow_html=True)
    st.markdown("**Top 10 Most Popular Tracks**")
    top_songs = df.nlargest(10, 'popularity')[['name', 'artists', 'popularity', 'year']]
    fig = px.bar(
        top_songs, x='popularity', y='name',
        color='popularity',
        color_continuous_scale=[[0, "#C7D2FE"], [1, "#4F46E5"]],
        orientation='h', title="", text='popularity'
    )
    fig.update_layout(**_base_layout(
        height=420, yaxis_title="", xaxis_title="Popularity",
    ))
    fig.update_traces(textposition='inside', textfont_color='white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close(), unsafe_allow_html=True)


def plot_correlation_analysis(df):
    """Correlation heatmap + scatter plots."""
    st.markdown('<div class="section-title">📊 Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    numeric_features = df[['popularity', 'danceability', 'energy', 'loudness',
                            'acousticness', 'valence', 'tempo', 'duration']]
    corr_matrix = numeric_features.corr(method='pearson')

    # Heatmap
    st.markdown(card_open(), unsafe_allow_html=True)
    st.markdown("**Feature Correlation Heatmap**")
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale=[[0, "#4F46E5"], [0.5, "#F5F7FA"], [1, "#0EA5E9"]],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont={"size": 11, "color": TEXT_PRIMARY},
        colorbar=dict(title="r")
    ))
    fig.update_layout(**_base_layout(height=480))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close(), unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Scatter pair
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Energy vs Loudness**")
        sample = df.sample(min(500, len(df)))
        fig = px.scatter(
            sample, x='energy', y='loudness',
            color='popularity',
            color_continuous_scale=[[0, "#E0E7FF"], [0.5, "#818CF8"], [1, "#4F46E5"]],
            opacity=0.7, title=""
        )
        fig.update_layout(**_base_layout(height=360, xaxis_title="Energy", yaxis_title="Loudness"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    with col2:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Acousticness vs Popularity**")
        sample = df.sample(min(500, len(df)))
        fig = px.scatter(
            sample, x='acousticness', y='popularity',
            color='energy',
            color_continuous_scale=[[0, "#ECFDF5"], [0.5, "#0EA5E9"], [1, "#1E293B"]],
            opacity=0.7, title=""
        )
        fig.update_layout(**_base_layout(height=360, xaxis_title="Acousticness", yaxis_title="Popularity"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)


def plot_time_series_analysis(df):
    """Yearly trends."""
    st.markdown('<div class="section-title">📅 Time Series Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    yearly_stats = get_yearly_stats(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Tracks per Year**")
        fig = px.bar(
            yearly_stats, x='year', y='歌曲数量',
            color='歌曲数量',
            color_continuous_scale=[[0, "#C7D2FE"], [1, "#4F46E5"]],
            title=""
        )
        fig.update_layout(**_base_layout(height=360, xaxis_title="Year", yaxis_title="Track Count"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    with col2:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Avg Duration Over Time**")
        fig = px.line(yearly_stats, x='year', y='平均时长', title="")
        fig.update_traces(line_color=ACCENT, line_width=3)
        fig.update_layout(**_base_layout(height=360, xaxis_title="Year", yaxis_title="Avg Duration (s)"))
        fig.add_scatter(
            x=yearly_stats['year'], y=yearly_stats['平均时长'],
            mode='markers', marker=dict(size=7, color=ACCENT),
            name='', showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Popularity over time
    st.markdown(card_open(), unsafe_allow_html=True)
    st.markdown("**Popularity Trend Over Time**")
    fig = px.line(yearly_stats, x='year', y='平均流行度', title="")
    fig.update_traces(line_color="#0EA5E9", line_width=3)
    fig.update_layout(**_base_layout(height=300, xaxis_title="Year", yaxis_title="Avg Popularity"))
    fig.add_scatter(
        x=yearly_stats['year'], y=yearly_stats['平均流行度'],
        mode='markers', marker=dict(size=7, color="#0EA5E9"),
        name='', showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close(), unsafe_allow_html=True)


def plot_genre_analysis(sp_feature):
    """Genre breakdown charts."""
    st.markdown('<div class="section-title">🎼 Genre Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Avg Duration by Genre**")
        genre_duration = sp_feature.groupby('genre')['duration'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=genre_duration.values, y=genre_duration.index,
            orientation='h', color=genre_duration.values,
            color_continuous_scale=[[0, "#C7D2FE"], [1, "#4F46E5"]],
            title=""
        )
        fig.update_layout(**_base_layout(height=400, xaxis_title="Avg Duration (s)", yaxis_title=""))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    with col2:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Danceability by Genre**")
        genre_dance = sp_feature.groupby('genre')['danceability'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=genre_dance.values, y=genre_dance.index,
            orientation='h', color=genre_dance.values,
            color_continuous_scale=[[0, "#BAE6FD"], [1, "#0EA5E9"]],
            title=""
        )
        fig.update_layout(**_base_layout(height=400, xaxis_title="Avg Danceability", yaxis_title=""))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Genre distribution donut
    st.markdown(card_open(), unsafe_allow_html=True)
    st.markdown("**Genre Distribution**")
    genre_counts = sp_feature['genre'].value_counts()
    fig = px.pie(
        values=genre_counts.values, names=genre_counts.index,
        hole=0.45, color_discrete_sequence=CLUSTER_COLORS
    )
    fig.update_layout(**_base_layout(height=420))
    fig.update_traces(textfont_color="white", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(card_close(), unsafe_allow_html=True)


def plot_kmeans_analysis(df, n_clusters):
    """K-Means cluster scatter + distribution."""
    st.markdown('<div class="section-title">🧠 K-Means Cluster Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    feature_cols = ['danceability', 'energy', 'loudness', 'acousticness',
                    'valence', 'tempo']
    labels, idx = run_kmeans(df, n_clusters, feature_cols)

    if labels is None:
        st.warning("Not enough data for K-Means clustering with the current filters.")
        return

    cluster_df = df.loc[idx].copy()
    cluster_df['Cluster'] = labels.astype(str)

    color_map = _cluster_color_map(n_clusters)

    # Layout: 2/3 scatter, 1/3 distribution
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Cluster Scatter — Danceability vs Energy**")
        x_col = 'danceability' if 'danceability' in cluster_df.columns else 'duration'
        y_col = 'energy' if 'energy' in cluster_df.columns else 'popularity'
        sample = cluster_df.sample(min(1500, len(cluster_df)))
        fig = px.scatter(
            sample, x=x_col, y=y_col, color='Cluster',
            color_discrete_map=color_map,
            opacity=0.75, title=""
        )
        fig.update_layout(**_base_layout(
            height=460,
            xaxis_title="Danceability" if x_col == 'danceability' else "Duration",
            yaxis_title="Energy" if y_col == 'energy' else "Popularity",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    with col_side:
        st.markdown(card_open(), unsafe_allow_html=True)
        st.markdown("**Cluster Size Distribution**")
        counts = cluster_df['Cluster'].value_counts().sort_index()
        fig = px.bar(
            x=counts.index, y=counts.values,
            color=counts.index, color_discrete_map=color_map,
            title=""
        )
        fig.update_layout(**_base_layout(
            height=460, xaxis_title="Cluster", yaxis_title="Tracks",
            showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(card_close(), unsafe_allow_html=True)


def show_overview_stats(df, sp_feature):
    """Overview stats section with additional metric row."""
    st.markdown('<div class="section-title">📈 Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    total_genres = len(sp_feature['genre'].unique()) if 'genre' in sp_feature.columns else 0
    years_range = f"{df['year'].min()} – {df['year'].max()}" if len(df) else "—"
    most_freq_genre = "—"
    if len(sp_feature) and 'genre' in sp_feature.columns:
        mode_vals = sp_feature['genre'].mode()
        if len(mode_vals):
            most_freq_genre = str(mode_vals.iloc[0])

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("📅", years_range, "Year Range", "Span of release dates")
    with c2:
        render_metric_card("🎸", f"{total_genres}", "Genres", "Unique genre categories")
    with c3:
        render_metric_card("👑", most_freq_genre, "Top Genre", "Most frequent genre in dataset")


def show_data_table(df):
    """Bottom row: full-width data preview table."""
    st.markdown('<div class="section-title">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown(card_open(), unsafe_allow_html=True)
    display_cols = [c for c in ['name', 'artists', 'popularity', 'danceability',
                                 'energy', 'valence', 'tempo', 'duration', 'year']
                    if c in df.columns]
    preview = df[display_cols].head(100).copy()
    preview.index = range(1, len(preview) + 1)
    st.dataframe(
        preview,
        use_container_width=True,
        height=420,
    )
    st.markdown(card_close(), unsafe_allow_html=True)


# ============================================================================
# Main Application
# ============================================================================
def main():
    """Main entry point."""
    sp_tracks, sp_feature = load_and_process_data()

    # ── Sidebar ──
    st.sidebar.markdown("## 🎵 Spotify Analytics")
    st.sidebar.markdown("---")

    min_year = int(sp_tracks['year'].min())
    max_year = int(sp_tracks['year'].max())
    year_range = st.sidebar.slider(
        "Year Range", min_value=min_year, max_value=max_year,
        value=(min_year, max_year), step=1
    )

    n_clusters = st.sidebar.slider(
        "K-Means Clusters (K)", min_value=2, max_value=10, value=3, step=1,
        help="Number of clusters for K-Means segmentation"
    )

    genres = ['全部类型'] + sorted(sp_feature['genre'].unique().tolist())
    selected_genres = st.sidebar.multiselect(
        "Genre Filter", genres, default=['全部类型']
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analysis Views")
    analysis_options = st.sidebar.multiselect(
        "Select views to display",
        ["Overview", "Popularity", "K-Means Clustering",
         "Correlation", "Time Series", "Genre Analysis", "Data Table"],
        default=["Overview", "Popularity", "K-Means Clustering", "Data Table"]
    )

    # ── Filter data ──
    filtered_tracks = filter_by_year(sp_tracks, year_range)
    filtered_features = filter_by_genre(sp_feature, selected_genres)

    # ── Page Header ──
    st.markdown(f"""
    <div class="page-header">
        <h1>🎵 Spotify Analytics Dashboard</h1>
        <p class="subtitle">
            Exploring <strong>{len(filtered_tracks):,}</strong> tracks
            from <strong>{year_range[0]}</strong> to <strong>{year_range[1]}</strong>
            {f' &nbsp;·&nbsp; Genres: {", ".join(selected_genres)}' if '全部类型' not in selected_genres and len(selected_genres) > 0 else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Row: 3 Metric Cards ──
    render_dashboard_metrics(filtered_tracks, filtered_features)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ── Analysis Sections ──
    if "Overview" in analysis_options:
        show_overview_stats(filtered_tracks, filtered_features)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "Popularity" in analysis_options:
        plot_popularity_analysis(filtered_tracks)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "K-Means Clustering" in analysis_options:
        plot_kmeans_analysis(filtered_tracks, n_clusters)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "Correlation" in analysis_options:
        plot_correlation_analysis(filtered_tracks)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "Time Series" in analysis_options:
        plot_time_series_analysis(filtered_tracks)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "Genre Analysis" in analysis_options:
        plot_genre_analysis(filtered_features)
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    if "Data Table" in analysis_options:
        show_data_table(filtered_tracks)

    # ── Footer ──
    st.markdown(f"""
    <div class="app-footer">
        Spotify Analytics Dashboard &nbsp;·&nbsp; Analyzing {len(filtered_tracks):,} tracks
        &nbsp;·&nbsp; Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
