"""
App Configuration & Custom CSS
================================
Page setup and premium dark theme styling for the AI Data Analyst.
"""

import streamlit as st


def set_page_config():
    """Sets the browser tab title and wide layout."""
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css():
    """Injects premium, responsive CSS for a professional dark theme."""
    st.markdown("""
        <style>
        /* ── Import Modern Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Color Palette ── */
        :root {
            --deep-blue: #0A192F;
            --slate: #8892B0;
            --light-slate: #a8b2d1;
            --accent: #64ffda;
            --accent-dim: rgba(100, 255, 218, 0.15);
            --bg: #020c1b;
            --box-bg: #112240;
            --border-glow: rgba(100, 255, 218, 0.3);
            --danger: #FF6B6B;
            --warning: #FBBF24;
            --info: #4D96FF;
        }

        /* ── Main Background ── */
        .stApp {
            background-color: var(--bg);
            color: var(--light-slate);
            font-family: 'Inter', sans-serif;
        }

        /* ── Container ── */
        .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px;
        }

        /* ── Typography ── */
        h1, h2, h3 {
            color: var(--accent) !important;
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.02em;
        }
        h1 { font-weight: 700 !important; }
        h2 { font-weight: 600 !important; }
        h3 { font-weight: 500 !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: var(--deep-blue);
            border-radius: 10px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--slate) !important;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-dim) !important;
            color: var(--accent) !important;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent) !important;
            background-color: rgba(100, 255, 218, 0.05);
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.5rem;
        }

        /* ── Buttons ── */
        .stButton>button {
            background-color: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--accent) !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stButton>button:hover {
            background-color: var(--accent-dim) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(100, 255, 218, 0.2) !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: var(--deep-blue) !important;
            border-right: 1px solid var(--box-bg);
        }
        [data-testid="stSidebar"] .block-container {
            padding: 1rem 1.5rem !important;
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }

        /* ── Chat Messages ── */
        [data-testid="stChatMessage"] {
            background-color: var(--box-bg) !important;
            border-radius: 10px !important;
            border-left: 3px solid var(--accent) !important;
            margin-bottom: 10px;
            padding: 12px 16px !important;
        }

        /* ── DataFrames ── */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--box-bg);
            border-radius: 8px;
        }

        /* ── Chart Containers ── */
        .stPlotlyChart {
            background-color: var(--box-bg) !important;
            border-radius: 12px !important;
            padding: 10px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
            border: 1px solid var(--border-glow) !important;
            margin-bottom: 16px !important;
        }

        /* ── Metrics ── */
        [data-testid="stMetricValue"] {
            color: var(--accent) !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--slate) !important;
        }
        [data-testid="metric-container"] {
            background-color: var(--box-bg);
            border: 1px solid rgba(100, 255, 218, 0.1);
            border-radius: 10px;
            padding: 16px !important;
        }

        /* ── Expanders ── */
        .streamlit-expanderHeader {
            background-color: var(--box-bg) !important;
            border-radius: 8px !important;
            color: var(--accent) !important;
            font-weight: 500 !important;
        }

        /* ── Info/Warning/Error boxes ── */
        .stAlert {
            border-radius: 8px !important;
        }

        /* ── Spinner ── */
        .stSpinner > div {
            border-top-color: var(--accent) !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            border: 1px dashed var(--border-glow) !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }

        /* ── Radio buttons ── */
        .stRadio > div {
            gap: 8px;
        }
        .stRadio label {
            background-color: var(--box-bg) !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            border: 1px solid rgba(100, 255, 218, 0.1) !important;
            transition: all 0.2s ease !important;
        }
        .stRadio label:hover {
            border-color: var(--accent) !important;
        }

        /* ── JSON viewer ── */
        .stJson {
            background-color: var(--box-bg) !important;
            border-radius: 8px !important;
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea {
            background-color: var(--box-bg) !important;
            border: 1px solid var(--border-glow) !important;
            border-radius: 10px !important;
            color: var(--light-slate) !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--deep-blue);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--slate);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }

        /* ── Hide Streamlit branding ── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)