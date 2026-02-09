# Spotify 数据分析仪表板

交互式 Streamlit 网页应用，用于分析 Spotify 音乐数据趋势和特征。

## 功能特性

- **年份筛选**: 通过滑块选择 1900-2020 年的年份范围
- **音乐类型筛选**: 多选框选择特定音乐类型或查看全部类型
- **数据概览**: 显示总歌曲数、平均流行度、平均时长等统计信息
- **流行度分析**:
  - 最受欢迎的歌曲排名
  - 流行度分布直方图
- **相关性分析**:
  - 特征相关性热力图
  - 能量 vs 响度散点图
  - 流行度 vs 声学特征散点图
- **时间序列分析**:
  - 各年份歌曲数量分布
  - 平均时长随年份变化
  - 流行度随时间变化趋势
- **音乐类型分析**:
  - 各类型平均时长对比
  - 各类型流行度排名
  - 音乐类型分布饼图

## 技术栈

- **Streamlit** - 网页应用框架
- **Pandas** - 数据处理
- **Plotly** - 交互式数据可视化
- **NumPy** - 数值计算
- **Matplotlib/Seaborn** - 静态图表支持

## 安装步骤

### 1. 创建虚拟环境（可选）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install streamlit plotly pandas numpy matplotlib seaborn
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址为 `http://localhost:8501`

## 项目结构

```
Spotify-Analysis/
├── app.py                  # 主应用文件
├── data/                   # 数据目录
│   ├── tracks.csv         # 音轨数据
│   └── SpotifyFeatures.csv # 特征数据
├── venv/                  # 虚拟环境（可选）
├── README.md              # 说明文档
└── spotify.py             # 原始分析脚本
```

## 数据说明

### tracks.csv
包含歌曲的基本信息和音频特征：

| 字段 | 说明 |
|------|------|
| id | 歌曲ID |
| name | 歌曲名称 |
| artists | 艺术家 |
| popularity | 流行度 (0-100) |
| release_date | 发行日期 |
| duration | 时长（秒） |
| explicit | 是否包含敏感内容 |
| key | 音调 (0-11) |
| mode | 调式 (0=小调, 1=大调) |
| tempo | 速度 (BPM) |
| danceability | 可舞性 |
| energy | 能量 |
| loudness | 响度 |
| acousticness | 声学特征 |
| instrumentalness | 器乐特征 |
| valence | 情感值 |

### SpotifyFeatures.csv
包含不同音乐类型的统计特征：

| 字段 | 说明 |
|------|------|
| genre | 音乐类型 |
| danceability | 平均可舞性 |
| energy | 平均能量 |
| key | 平均音调 |
| loudness | 平均响度 |
| mode | 平均调式 |
| speechiness | 平均语音度 |
| acousticness | 平均声学特征 |
| instrumentalness | 平均器乐特征 |
| liveness | 平均现场感 |
| valence | 平均情感值 |
| tempo | 平均速度 |
| duration | 平均时长 |

## 使用指南

1. **侧边栏控制**:
   - 使用年份滑块选择时间范围
   - 使用多选框选择音乐类型
   - 勾选想要查看的分析视图

2. **交互功能**:
   - 所有图表都支持鼠标悬停查看详细信息
   - 图表可以缩放和平移
   - 点击图例可以隐藏/显示数据系列

3. **数据筛选**:
   - 年份筛选会实时更新所有图表
   - 音乐类型筛选只影响音乐类型分析部分

## 性能优化

- 使用 Streamlit 的 `@st.cache_data` 装饰器缓存数据加载
- 大数据集时自动采样以提高渲染速度
- 预计算常用的统计指标

## 部署选项

### 本地部署
```bash
streamlit run app.py
```

### Streamlit Cloud
1. 将代码推送到 GitHub
2. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
3. 连接 GitHub 仓库并部署

### Heroku
需要创建 `requirements.txt` 和 `Procfile`:

**Procfile**:
```
web: streamlit run app.py --server.port=$PORT
```

## 常见问题

**Q: 应用启动很慢？**
A: 首次运行会加载数据和创建缓存，后续运行会更快。

**Q: 图表显示不完整？**
A: 尝试调整浏览器窗口大小或缩放页面。

**Q: 中文显示乱码？**
A: 确保系统已安装中文字体，或修改 `plt.rcParams` 中的字体设置。

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue。
