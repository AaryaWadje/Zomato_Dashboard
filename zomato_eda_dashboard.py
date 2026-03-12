import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato EDA Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0f0f0f;
        color: #f0ece4;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1208 50%, #0f0f0f 100%);
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #f5a623 !important;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 900;
        color: #f5a623;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #c9b99a;
        font-weight: 300;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e1a14, #2a2318);
        border: 1px solid #3d3020;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(245,166,35,0.08);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #f5a623;
    }

    .metric-number {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f5a623;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #a89070;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 0.2rem;
    }

    .chart-caption {
        font-size: 0.88rem;
        color: #a89070;
        font-style: italic;
        line-height: 1.6;
        padding: 0.6rem 0.2rem 1.2rem;
        border-top: 1px solid #2a2318;
        margin-top: 0.4rem;
    }

    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #f5a623;
        margin-bottom: 0.3rem;
    }

    div[data-testid="stSidebar"] {
        background: #130f09 !important;
        border-right: 1px solid #2a2318;
    }

    div[data-testid="stSidebar"] * {
        color: #c9b99a !important;
    }

    .stSelectbox label, .stMultiSelect label {
        color: #a89070 !important;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }

    .separator {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3d3020, transparent);
        margin: 2rem 0;
    }

    .insight-box {
        background: linear-gradient(135deg, #1e1a14, #261f14);
        border-left: 3px solid #f5a623;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1.2rem;
    }

    .insight-box p {
        margin: 0;
        font-size: 0.9rem;
        color: #c9b99a;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ──────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#f5a623',
    'secondary': '#e8643a',
    'accent': '#c9b99a',
    'bg': '#1a1208',
    'card': '#1e1a14',
    'grid': '#2a2318',
    'text': '#f0ece4',
    'palette': ['#f5a623', '#e8643a', '#d4a853', '#c96a3e', '#b8954a', '#a07840',
                '#f0c060', '#e07030', '#c8b040', '#d06835']
}

def apply_theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color=COLORS['text'], size=12),
        title=dict(text=title, font=dict(family='Playfair Display', size=18, color=COLORS['primary']),
                   x=0.02, xanchor='left'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=COLORS['accent'])),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'],
                   tickfont=dict(color=COLORS['accent'])),
        yaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'],
                   tickfont=dict(color=COLORS['accent']))
    )
    return fig

# ─── DATA LOADER ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Clean rate column
    if df['rate'].dtype == object:
        df['rate'] = df['rate'].astype(str).str.replace('/5', '').str.strip()
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

    # Clean cost column
    cost_col = [c for c in df.columns if 'cost' in c.lower()]
    if cost_col:
        df[cost_col[0]] = df[cost_col[0]].astype(str).str.replace(',', '').str.strip()
        df[cost_col[0]] = pd.to_numeric(df[cost_col[0]], errors='coerce')
        df.rename(columns={cost_col[0]: 'cost_for_two'}, inplace=True)

    # Fill missing
    if 'rate' in df.columns:
        df['rate'].fillna(df['rate'].mean(), inplace=True)
    if 'cost_for_two' in df.columns:
        df['cost_for_two'].fillna(df['cost_for_two'].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['name', 'location'], inplace=True)

    # Encode
    for col in ['online_order', 'book_table']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map({'yes': 1, 'no': 0, '1': 1, '0': 0})

    return df

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍽️ Zomato EDA")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your cleaned CSV", type=['csv'])
    st.markdown("---")
    st.markdown("**Navigation**")
    sections = ["Overview", "Ratings Analysis", "Location Insights",
                "Cuisine Deep Dive", "Cost & Value", "Delivery & Booking"]
    selected = st.radio("", sections, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<span style='font-size:0.75rem;color:#5a5040;'>Zomato Bangalore EDA · 2024</span>", unsafe_allow_html=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Landing screen
    st.markdown("""
    <div style='text-align:center; padding: 5rem 2rem;'>
        <div class='hero-title'>Zomato Restaurant<br>Analytics Dashboard</div>
        <div class='hero-sub' style='margin-top:1rem;'>Upload your cleaned Zomato CSV to begin exploration</div>
        <div style='margin-top:3rem; font-size:4rem;'>🍽️</div>
        <div style='margin-top:2rem; color:#5a5040; font-size:0.85rem;'>
            Use the sidebar to upload your <b style='color:#a89070;'>zomato_cleaned.csv</b> file
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = load_data(uploaded_file)

# ── FILTERS ──
if selected != "Overview":
    with st.sidebar:
        st.markdown("**Filters**")
        if 'location' in df.columns:
            top_locs = df['location'].value_counts().head(20).index.tolist()
            sel_locs = st.multiselect("Locations", top_locs, default=top_locs[:8])
            if sel_locs:
                df = df[df['location'].isin(sel_locs)]
        if 'rate' in df.columns:
            min_r, max_r = float(df['rate'].min()), float(df['rate'].max())
            r_range = st.slider("Rating Range", min_r, max_r, (min_r, max_r), 0.1)
            df = df[(df['rate'] >= r_range[0]) & (df['rate'] <= r_range[1])]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if selected == "Overview":
    st.markdown("<div class='hero-title'>Zomato Bangalore</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Exploratory Data Analysis · Restaurant Intelligence Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(df):,}", "Total Restaurants"),
        (f"{df['location'].nunique() if 'location' in df.columns else '—'}", "Locations"),
        (f"{round(df['rate'].mean(), 2) if 'rate' in df.columns else '—'}", "Avg Rating"),
        (f"₹{int(df['cost_for_two'].median()) if 'cost_for_two' in df.columns else '—'}", "Median Cost (2)"),
        (f"{df['cuisines'].nunique() if 'cuisines' in df.columns else '—'}", "Cuisine Types"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number'>{num}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Rating distribution
    with col1:
        st.markdown("<div class='section-label'>Rating Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x='rate', nbins=30, color_discrete_sequence=[COLORS['primary']])
        fig.update_traces(marker_line_color='#0f0f0f', marker_line_width=1, opacity=0.9)
        fig = apply_theme(fig, "How Are Restaurants Rated?")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        Most restaurants cluster between 3.5 and 4.2 stars, suggesting a competitive mid-tier market.
        Very few restaurants score above 4.5, making them standout performers worth studying.
        </div>""", unsafe_allow_html=True)

    # Restaurant type donut
    with col2:
        if 'rest_type' in df.columns:
            st.markdown("<div class='section-label'>Restaurant Types</div>", unsafe_allow_html=True)
            type_counts = df['rest_type'].value_counts().head(8)
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                         hole=0.55, color_discrete_sequence=COLORS['palette'])
            fig.update_traces(textfont_color='white', pull=[0.05] + [0]*7)
            fig = apply_theme(fig, "Restaurant Type Breakdown")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            Quick Bites and Casual Dining dominate the Bangalore food scene, reflecting urban demand for fast, affordable meals.
            Fine Dining represents only a small fraction, indicating a price-sensitive consumer base.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: RATINGS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Ratings Analysis":
    st.markdown("<div class='hero-title'>Ratings Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Understanding what drives customer satisfaction</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Ratings vs Cost scatter
    with col1:
        st.markdown("<div class='section-label'>Rating vs Cost</div>", unsafe_allow_html=True)
        fig = px.scatter(df.sample(min(3000, len(df))), x='cost_for_two', y='rate',
                         color='rate', color_continuous_scale=['#e8643a', '#f5a623', '#d4e157'],
                         opacity=0.6, size_max=6)
        fig.update_traces(marker=dict(size=5))
        fig = apply_theme(fig, "Does Higher Cost Mean Better Rating?")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        There is no strong linear relationship between price and rating — affordable restaurants often match expensive ones in satisfaction.
        This indicates that value-for-money and food quality matter more than price positioning alone.
        </div>""", unsafe_allow_html=True)

    # Online order vs ratings box
    with col2:
        if 'online_order' in df.columns:
            st.markdown("<div class='section-label'>Online Order vs Rating</div>", unsafe_allow_html=True)
            df_box = df.copy()
            df_box['Online Order'] = df_box['online_order'].map({1: 'Yes', 0: 'No'})
            fig = px.box(df_box, x='Online Order', y='rate',
                         color='Online Order',
                         color_discrete_map={'Yes': COLORS['primary'], 'No': COLORS['secondary']})
            fig = apply_theme(fig, "Do Online Order Restaurants Rate Higher?")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            Restaurants offering online ordering tend to have slightly higher median ratings, possibly due to broader customer reach and feedback.
            The wider spread in non-online restaurants suggests inconsistent quality in traditional dine-in only setups.
            </div>""", unsafe_allow_html=True)

    # Top rated restaurants
    st.markdown("<div class='section-label'>Top Rated Restaurants</div>", unsafe_allow_html=True)
    if 'votes' in df.columns:
        top = df[df['votes'] > 50].nlargest(15, 'rate')[['name', 'location', 'rate', 'votes', 'cost_for_two']].drop_duplicates('name')
        fig = px.bar(top, x='rate', y='name', orientation='h',
                     color='votes', color_continuous_scale=['#3d3020', '#f5a623'],
                     text='rate')
        fig.update_traces(textfont_color='white', textposition='outside')
        fig = apply_theme(fig, "Top 15 Highly Voted Restaurants")
        fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        These restaurants combine high ratings with substantial vote counts, making them statistically reliable top performers.
        Their locations and cuisine types offer a blueprint for launching a successful new food venture in Bangalore.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: LOCATION INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Location Insights":
    st.markdown("<div class='hero-title'>Location Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Where does Bangalore eat?</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-label'>Restaurant Density by Location</div>", unsafe_allow_html=True)
        loc_counts = df['location'].value_counts().head(20).reset_index()
        loc_counts.columns = ['location', 'count']
        fig = px.bar(loc_counts, x='count', y='location', orientation='h',
                     color='count', color_continuous_scale=['#3d3020', COLORS['primary'], '#ffffff'])
        fig = apply_theme(fig, "Top 20 Locations by Restaurant Count")
        fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=550)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        BTM, Koramangala, and Indiranagar lead with the highest concentration of restaurants, reflecting dense urban neighborhoods.
        Identifying underrepresented areas with fewer restaurants can reveal untapped market opportunities for new entrants.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-label'>Average Rating by Location</div>", unsafe_allow_html=True)
        loc_rating = df.groupby('location')['rate'].mean().sort_values(ascending=False).head(20).reset_index()
        fig = px.bar(loc_rating, x='rate', y='location', orientation='h',
                     color='rate', color_continuous_scale=['#e8643a', '#f5a623', '#d4e157'])
        fig = apply_theme(fig, "Which Locations Have the Best Ratings?")
        fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=550)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        Upscale neighborhoods like Lavelle Road and Church Street consistently score higher ratings, driven by premium dining experiences.
        A cloud kitchen targeting high-rating zones should focus on quality and presentation to meet elevated customer expectations.
        </div>""", unsafe_allow_html=True)

    # Cost by location
    st.markdown("<div class='section-label'>Average Cost for Two by Location</div>", unsafe_allow_html=True)
    loc_cost = df.groupby('location')['cost_for_two'].mean().sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(loc_cost, x='location', y='cost_for_two',
                 color='cost_for_two', color_continuous_scale=['#3d3020', COLORS['primary']])
    fig = apply_theme(fig, "Most Expensive Areas to Dine In Bangalore")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    Areas like Lavelle Road and MG Road command the highest average spend per meal, correlating with wealthier demographics.
    For a budget cloud kitchen, targeting mid-cost zones like Whitefield or Electronic City offers a larger addressable market.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: CUISINE DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Cuisine Deep Dive":
    st.markdown("<div class='hero-title'>Cuisine Deep Dive</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>What does Bangalore love to eat?</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    # Explode multi-cuisine
    df_cuisine = df.copy()
    if 'cuisines' in df_cuisine.columns:
        df_cuisine = df_cuisine.dropna(subset=['cuisines'])
        df_cuisine['cuisine_list'] = df_cuisine['cuisines'].str.split(',')
        df_exploded = df_cuisine.explode('cuisine_list')
        df_exploded['cuisine_list'] = df_exploded['cuisine_list'].str.strip()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-label'>Most Popular Cuisines</div>", unsafe_allow_html=True)
            top_cuisines = df_exploded['cuisine_list'].value_counts().head(15).reset_index()
            top_cuisines.columns = ['cuisine', 'count']
            fig = px.bar(top_cuisines, x='count', y='cuisine', orientation='h',
                         color='count', color_continuous_scale=['#3d3020', COLORS['primary']])
            fig = apply_theme(fig, "Top 15 Cuisines in Bangalore")
            fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=500)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            North Indian and Chinese cuisines dominate the Bangalore food scene, reflecting widespread consumer familiarity.
            For a new cloud kitchen, these cuisines offer the largest built-in customer base with proven demand.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-label'>Cuisine vs Average Rating</div>", unsafe_allow_html=True)
            cuisine_rating = df_exploded.groupby('cuisine_list')['rate'].mean().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(cuisine_rating, x='rate', y='cuisine_list', orientation='h',
                         color='rate', color_continuous_scale=['#e8643a', '#f5a623', '#d4e157'])
            fig = apply_theme(fig, "Highest Rated Cuisines")
            fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=500)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            Specialty and niche cuisines like Mughlai and Biryani receive consistently high ratings despite lower volume.
            This signals a gap in the market — quality niche cuisines are appreciated but undersupplied.
            </div>""", unsafe_allow_html=True)

        # Treemap
        st.markdown("<div class='section-label'>Cuisine Landscape</div>", unsafe_allow_html=True)
        top20_c = df_exploded['cuisine_list'].value_counts().head(20).reset_index()
        top20_c.columns = ['cuisine', 'count']
        fig = px.treemap(top20_c, path=['cuisine'], values='count',
                         color='count', color_continuous_scale=['#2a1f0a', '#f5a623'])
        fig = apply_theme(fig, "Cuisine Treemap — Size = Popularity")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        The treemap visually illustrates the dominance of a few key cuisines while revealing many smaller, underexplored categories.
        Launching in a smaller tile category could mean less competition while still serving a defined customer segment.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: COST & VALUE
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Cost & Value":
    st.markdown("<div class='hero-title'>Cost & Value</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Understanding pricing dynamics across Bangalore</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-label'>Cost Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df[df['cost_for_two'] < 2000], x='cost_for_two', nbins=40,
                           color_discrete_sequence=[COLORS['secondary']])
        fig.update_traces(marker_line_color='#0f0f0f', marker_line_width=1)
        fig = apply_theme(fig, "How Much Do Bangaloreans Spend?")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        The majority of restaurant visits fall between ₹200–₹600 for two people, pointing to a budget-conscious dining culture.
        This sweet spot is ideal for a cloud kitchen positioning itself as affordable yet quality-driven.
        </div>""", unsafe_allow_html=True)

    with col2:
        if 'rest_type' in df.columns:
            st.markdown("<div class='section-label'>Cost by Restaurant Type</div>", unsafe_allow_html=True)
            cost_type = df.groupby('rest_type')['cost_for_two'].median().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(cost_type, x='rest_type', y='cost_for_two',
                         color='cost_for_two', color_continuous_scale=['#3d3020', COLORS['primary']])
            fig = apply_theme(fig, "Median Cost by Restaurant Type")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            Fine Dining and Bars command the highest median spend, while Quick Bites and Cafes stay in the affordable range.
            This validates that restaurant type is a strong predictor of pricing strategy and target customer segment.
            </div>""", unsafe_allow_html=True)

    # Votes vs Cost
    st.markdown("<div class='section-label'>Votes vs Cost Correlation</div>", unsafe_allow_html=True)
    if 'votes' in df.columns:
        fig = px.scatter(df[df['cost_for_two'] < 3000].sample(min(2000, len(df))),
                         x='cost_for_two', y='votes', color='rate',
                         color_continuous_scale=['#e8643a', '#f5a623', '#d4e157'],
                         opacity=0.6, size='rate', size_max=10)
        fig = apply_theme(fig, "Are Cheaper Restaurants More Popular?")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        Lower to mid-priced restaurants accumulate more votes, suggesting wider customer engagement at affordable price points.
        High vote counts with moderate pricing indicate strong community favorites — a winning formula for new cloud kitchens.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: DELIVERY & BOOKING
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == "Delivery & Booking":
    st.markdown("<div class='hero-title'>Delivery & Booking</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>How Bangalore orders and reserves</div>", unsafe_allow_html=True)
    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    online_pct = df['online_order'].mean() * 100 if 'online_order' in df.columns else 0
    book_pct = df['book_table'].mean() * 100 if 'book_table' in df.columns else 0
    both_pct = ((df['online_order'] == 1) & (df['book_table'] == 1)).mean() * 100 if all(c in df.columns for c in ['online_order', 'book_table']) else 0

    for col, (num, label) in zip([col1, col2, col3], [
        (f"{online_pct:.1f}%", "Offer Online Order"),
        (f"{book_pct:.1f}%", "Offer Table Booking"),
        (f"{both_pct:.1f}%", "Offer Both")
    ]):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number'>{num}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-label'>Online Order Adoption</div>", unsafe_allow_html=True)
        online_counts = df['online_order'].map({1: 'Accepts Online Order', 0: 'No Online Order'}).value_counts()
        fig = px.pie(values=online_counts.values, names=online_counts.index, hole=0.6,
                     color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
        fig.update_traces(pull=[0.05, 0])
        fig = apply_theme(fig, "Online Order Availability")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        A significant proportion of restaurants now accept online orders, reflecting the surge in food delivery culture post-pandemic.
        For a new cloud kitchen startup, offering online ordering from day one is a non-negotiable competitive requirement.
        </div>""", unsafe_allow_html=True)

    with col2:
        if 'rest_type' in df.columns:
            st.markdown("<div class='section-label'>Online Order by Restaurant Type</div>", unsafe_allow_html=True)
            pivot = df.groupby('rest_type')['online_order'].mean().sort_values(ascending=False).head(10).reset_index()
            pivot['online_order_pct'] = pivot['online_order'] * 100
            fig = px.bar(pivot, x='online_order_pct', y='rest_type', orientation='h',
                         color='online_order_pct', color_continuous_scale=['#3d3020', COLORS['primary']])
            fig = apply_theme(fig, "Which Restaurant Types Go Online Most?")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis=dict(categoryorder='total ascending'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""<div class='chart-caption'>
            Delivery-focused formats like Quick Bites and Casual Dining have the highest online ordering rates.
            Fine Dining establishments lag in online adoption, preferring the in-person experience as their core value proposition.
            </div>""", unsafe_allow_html=True)

    # Grouped comparison
    st.markdown("<div class='section-label'>Rating by Online Order & Table Booking</div>", unsafe_allow_html=True)
    df_grouped = df.copy()
    df_grouped['Category'] = df_grouped['online_order'].map({1: 'Online', 0: 'No Online'})
    fig = px.violin(df_grouped, x='Category', y='rate', color='Category',
                    color_discrete_map={'Online': COLORS['primary'], 'No Online': COLORS['secondary']},
                    box=True, points=False)
    fig = apply_theme(fig, "Rating Distribution: Online vs Non-Online")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    Restaurants with online ordering show a tighter and slightly higher rating distribution, suggesting better customer service accountability.
    This violin plot reveals that digital presence correlates not just with accessibility but with overall quality standards.
    </div>""", unsafe_allow_html=True)
