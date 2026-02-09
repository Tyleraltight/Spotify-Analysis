[中文](README_zh.md) | English

# Spotify Data Analysis Dashboard

An interactive Streamlit web application for analyzing music data trends and audio features from Spotify's extensive music catalog.

---

## Project Overview

This project transforms traditional data analysis scripts into an interactive web-based visualization platform, enabling real-time exploration of musical patterns across decades. The application provides comprehensive insights into how music characteristics have evolved from 1900 to 2020, examining relationships between acoustic features, popularity metrics, and temporal trends.

Built as part of a data science portfolio, this project demonstrates proficiency in:
- Data preprocessing and cleaning using Pandas
- Interactive visualization with Plotly
- Web application development with Streamlit
- Statistical analysis and correlation studies

---

## Technical Implementation

### Data Processing Pipeline

The application employs a modular architecture separating concerns between data ingestion, transformation, and visualization:

```python
@st.cache_data
def load_and_process_data():
    """Load and preprocess CSV datasets with caching optimization."""
    sp_tracks = pd.read_csv('data/tracks.csv')
    sp_feature = pd.read_csv('data/SpotifyFeatures.csv')

    # Timestamp conversion and feature engineering
    sp_tracks['release_date'] = pd.to_datetime(sp_tracks['release_date'])
    sp_tracks['year'] = sp_tracks['release_date'].dt.year
    sp_tracks['duration'] = sp_tracks['duration_ms'].apply(lambda x: round(x / 1000))

    return sp_tracks, sp_feature
```

### Filtering Mechanisms

Real-time data filtering enables dynamic exploration:

```python
def filter_by_year(df, year_range):
    """Apply temporal constraints to dataset."""
    return df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

def filter_by_genre(df, genres):
    """Apply genre-based subset selection."""
    if '全部类型' in genres or len(genres) == 0:
        return df
    return df[df['genre'].isin(genres)]
```

### Performance Optimization

- **Caching Strategy**: Streamlit's `@st.cache_data` decorator minimizes redundant I/O operations
- **Sampling Algorithm**: Large datasets undergo intelligent sampling (typically 0.4%) to maintain rendering performance while preserving statistical significance
- **Precomputation**: Yearly statistics are computed once during initialization

---

## Key Features

### 1. Interactive Controls
- **Year Range Slider**: Select temporal windows from 1900-2020
- **Genre Multi-Select**: Filter by specific musical categories
- **View Toggles**: Customize dashboard layout

### 2. Popularity Analysis
- Top-ranked tracks by popularity score
- Distribution histogram with density estimation
- Real-time updates based on temporal filters

### 3. Correlation Analysis
- **Pearson Correlation Heatmap**: Matrix visualization of feature interdependencies
- **Scatter Plot Matrix**: Energy vs. Loudness, Popularity vs. Acousticness
- Interactive tooltips displaying precise measurements

### 4. Time Series Analysis
- Annual track count distribution
- Temporal evolution of average song duration
- Popularity trends over decades

### 5. Genre Analysis
- Comparative duration analysis across musical genres
- Popularity ranking by genre
- Categorical distribution visualization

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/Tyleraltight/Spotify-Analysis.git
cd Spotify-Analysis
```

2. **Create virtual environment** (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Launch the application**:
```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|----------|
| Web Framework | Streamlit | Rapid application development |
| Data Processing | Pandas | Data manipulation and cleaning |
| Visualization | Plotly | Interactive charting |
| Numerical Computing | NumPy | Array operations |
| Statistical Graphics | Matplotlib/Seaborn | Static plotting support |

---

## Data Schema

### tracks.csv

| Field | Description | Type |
|-------|-------------|------|
| id | Unique track identifier | int |
| name | Track title | str |
| artists | Performing artist(s) | str |
| popularity | Spotify popularity metric (0-100) | int |
| release_date | Publication timestamp | datetime |
| duration | Track length in seconds | int |
| explicit | Explicit content flag | bool |
| key | Musical key (0-11) | int |
| mode | Modality (0=minor, 1=major) | int |
| tempo | Tempo in BPM | float |
| danceability | Danceability score (0-1) | float |
| energy | Energy level (0-1) | float |
| loudness | Loudness in dB | float |
| acousticness | Acoustic characteristic (0-1) | float |
| instrumentalness | Instrumental presence (0-1) | float |
| valence | Musical positiveness (0-1) | float |

### SpotifyFeatures.csv

| Field | Description | Type |
|-------|-------------|------|
| genre | Musical category | str |
| danceability | Average danceability per genre | float |
| energy | Average energy per genre | float |
| key | Average key per genre | int |
| loudness | Average loudness per genre | float |
| mode | Average modality per genre | int |
| speechiness | Average speechiness per genre | float |
| acousticness | Average acousticness per genre | float |
| instrumentalness | Average instrumentalness per genre | float |
| liveness | Average liveness per genre | float |
| valence | Average valence per genre | float |
| tempo | Average tempo per genre | float |
| duration | Average duration per genre | float |

---

## Deployment Options

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect repository at [Streamlit Cloud](https://streamlit.io/cloud)
3. Automatic deployment upon commit

### Heroku
Create `requirements.txt` and `Procfile`:

**Procfile**:
```
web: streamlit run app.py --server.port=$PORT
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

For inquiries or contributions, please open an issue on GitHub.
