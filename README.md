[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Tyleraltight/Spotify-Analysis&main_file=app.py)

# Spotify Audio Analysis Dashboard

A professional, enterprise-grade data visualization platform for exploring the Spotify music catalog. This application transforms raw audio features into actionable insights through a sophisticated SaaS-style interface.

![Dashboard Preview](https://img.shields.io/badge/Style-SaaS%20Professional-4F46E5?style=flat-square) ![Tech](https://img.shields.io/badge/Built%20With-Streamlit%20%7C%20Plotly-FF4B4B?style=flat-square)

---

## 🎨 Professional SaaS Identity

The user interface has been engineered for maximum clarity and visual hierarchy, featuring a **polished light-mode aesthetic**:

- **Unified Design System**: Built on a soft off-white canvas (`#F5F7FA`) with high-contrast charcoal typography (`#2D3748`) using the **Inter** font stack.
- **Card-Based Architecture**: precise data encapsulation within white, double-shadowed cards (`.saas-card`) that provide depth and separation.
- **Harmonized Color Palette**: A professional cold-tone visualization scheme (Indigo, Sky Blue, Slate) ensures readability and accessibility across all charts.
- **Interactive "Ghost" Controls**: Custom CSS injection transforms standard Streamlit widgets into refined, minimalist input groups.

## 🧠 Core Analytical Features

### 1. K-Means Clustering Engine
Leverages Unsupervised Learning to segment tracks based on audio signatures.
- **Algorithm**: `sklearn.cluster.KMeans` using **Euclidean distance**.
- **Feature Scaling**: `StandardScaler` normalization of 6 key dimensions (Danceability, Energy, Loudness, Acousticness, Valence, Tempo).
- **Interactive Segmentation**: Dynamic control over cluster count ($K=2..10$) to explore different granularity levels.

### 2. Interactive Data Visualization
Powered by **Plotly**, enabling deep drill-down capabilities:
- **Popularity Distribution**: Dual-view analysis pairing scatter plots (Danceability vs. Energy) with density histograms.
- **Correlation Matrix**: Pearson correlation heatmaps to identify relationships between audio features.
- **Time-Series Analysis**: Longitudinal tracking of song duration and popularity trends from 1900 to 2020.
- **Genre Profiling**: Comparative breakdown of acoustic characteristics across musical genres.

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Custom CSS/HTML Injection)
- **Visualization**: Plotly Express & Graph Objects
- **Data Engineering**: Pandas (Vectorized operations)
- **Machine Learning**: Scikit-learn (K-Means, Preprocessing)
- **Runtime**: Python 3.9+

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tyleraltight/Spotify-Analysis.git
   cd Spotify-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
   The application will deploy locally at `http://localhost:8501`.

---

## 📂 Data Structure

The analysis is driven by two primary datasets located in the `data/` directory:

| Dataset | Description | Key Features |
| :--- | :--- | :--- |
| **tracks.csv** | Granular track-level data | `id`, `name`, `popularity`, `release_date`, `danceability`, `energy`, `loudness` |
| **SpotifyFeatures.csv** | Genre-aggregated statistics | `genre`, `acousticness`, `instrumentalness`, `tempo`, `key`, `mode` |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
