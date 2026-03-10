# 🤖 AI Autonomous Data Analyst

> **Upload any CSV or Excel file → Get instant, professional-grade data analysis with AI-powered insights.**

An intelligent data analysis platform built with **Streamlit** that automatically performs end-to-end exploratory data analysis (EDA), generates interactive visualizations, and provides an AI-powered conversational interface for querying your data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [How It Works (End-to-End Flow)](#-how-it-works-end-to-end-flow)
- [Modules Explained](#-modules-explained-in-simple-terms)
- [Models & Algorithms Used](#-models--algorithms-used)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📁 **Multi-format Upload** | Supports CSV, XLSX, and XLS files |
| 🧠 **Schema Intelligence** | Auto-detects data types, primary keys, targets, and time series |
| 🔍 **Data Quality Audit** | Finds missing values, duplicates, outliers, and format issues |
| 📊 **Statistical Profiling** | Descriptive stats, distributions, correlations, feature importance |
| 📈 **16+ Auto Visualizations** | Interactive Plotly charts with annotations and legends |
| 🤖 **AI Executive Summary** | LLM-generated insights, anomalies, and recommendations |
| 💬 **AI Consultant Chat** | Ask questions about your data in natural language |
| 🔄 **Dual LLM Support** | Ollama (local, free) with OpenRouter cloud fallback |
| 🎨 **Premium Dark UI** | Professional theme with glassmorphism and smooth animations |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)              │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐    │
│  │ Dashboard │  │ AI Consultant│  │  Sidebar    │    │
│  │  (5 tabs) │  │   (Chat UI)  │  │ (Upload +   │    │
│  │           │  │              │  │  Navigation)│    │
│  └─────┬─────┘  └──────┬───────┘  └──────┬──────┘    │
│        │               │                 │           │
├────────┼───────────────┼─────────────────┼───────────┤
│        ▼               ▼                 ▼           │
│  ┌──────────────────────────────────────────────┐    │
│  │           CONTEXT MEMORY (memory.py)          │    │
│  │  Session state manager for all analysis data  │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                │
├─────────────────────┼────────────────────────────────┤
│    ANALYSIS ENGINES  │                                │
│  ┌──────────┐ ┌──────┴─────┐ ┌──────────────┐       │
│  │ Schema   │ │ Data       │ │ Statistical  │       │
│  │ Intel    │ │ Quality    │ │ Profiler     │       │
│  └──────────┘ └────────────┘ └──────────────┘       │
│                                                      │
│  ┌──────────────┐  ┌───────────────────────────┐     │
│  │ Data Handler │  │ Visualization Engine      │     │
│  │ (CSV/Excel)  │  │ (16+ Plotly Charts)       │     │
│  └──────────────┘  └───────────────────────────┘     │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │          AI AGENT (ai_agent.py)               │    │
│  │  Ollama (local) ──► OpenRouter (cloud)        │    │
│  │  Executive Summary + Conversational Chat      │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI with dark theme |
| **Data Processing** | Pandas, NumPy | DataFrame manipulation, cleaning |
| **Visualization** | Plotly | Interactive, dark-themed charts |
| **Statistics** | SciPy, Scikit-learn | Normality tests, clustering, feature importance |
| **LLM (Local)** | Ollama (llama3.2:3b) | Free, private, no rate limits |
| **LLM (Cloud)** | OpenRouter (OpenAI SDK) | Fallback with multiple free models |
| **Styling** | Custom CSS | Premium glassmorphism dark theme |

---

## 🔄 How It Works (End-to-End Flow)

Here's exactly what happens when you upload a file, explained step by step:

### Step 1: File Upload & Loading
```
User uploads CSV/Excel → data_handler.py
```
- Detects file type from extension (`.csv`, `.xlsx`, `.xls`)
- For CSV: tries UTF-8 encoding first, falls back to Latin-1
- For Excel: uses `openpyxl` engine
- Returns a raw Pandas DataFrame

### Step 2: Data Cleaning
```
Raw DataFrame → data_handler.py → Clean DataFrame
```
- Strips whitespace from column names
- Converts column names to lowercase with underscores
- Drops completely empty rows
- Converts columns to optimal data types (downcasting numbers, parsing dates)
- Creates a mapping of original → cleaned column names

### Step 3: Schema Intelligence
```
Clean DataFrame → schema_intel.py → Schema Report
```
- **Type Detection**: Classifies each column as numeric, categorical, boolean, datetime, text, or identifier
- **Mixed Column Detection**: Finds columns with inconsistent data types
- **Primary Key Detection**: Identifies columns with 100% unique, non-null values
- **Target Detection**: Scores columns likely to be prediction targets (by name patterns like "target", "label", "status" + cardinality)
- **Time Series Detection**: Checks for datetime columns and detects frequency (daily, monthly, etc.)

### Step 4: Data Quality Audit  
```
Clean DataFrame → data_quality.py → Quality Report (0-100 score)
```
- **Missing Values**: Counts missing per column, classifies severity (low/moderate/high/critical)
- **Duplicates**: Detects exact duplicate rows and near-duplicates
- **Format Issues**: Finds whitespace problems, mixed case variants ("New York" vs "new york")
- **Outliers**: Uses IQR (Interquartile Range) method — values beyond Q1-1.5×IQR or Q3+1.5×IQR
- **Class Imbalance**: Checks if target columns have skewed class distributions
- **Quality Score**: Computes 0-100 score across 4 dimensions (completeness, uniqueness, consistency, validity), each worth 25 points

### Step 5: Statistical Profiling
```
Clean DataFrame → stats_profiler.py → Profiling Report
```
- **Descriptive Stats**: Mean, median, std, skewness, kurtosis for all numeric columns
- **Distribution Tests**: D'Agostino-Pearson normality test (from SciPy) to classify each column's distribution
- **Correlations**: Pearson correlation matrix + top positive/negative pairs
- **Feature Importance**: Mutual information scoring (if target found) or variance-based ranking
- **Clustering**: K-Means with automatic k selection via silhouette score

### Step 6: Context Building
```
All Reports → memory.py → LLM Context String
```
- Combines schema summary, quality summary, and profiling summary into a single text prompt
- Stores everything in Streamlit session state for persistence during the user's session
- Builds enriched context for follow-up AI queries

### Step 7: AI Executive Summary
```
LLM Context → ai_agent.py → Structured JSON Summary
```
- Sends the data context to **Ollama** (local LLM) as the primary provider
- If Ollama is unavailable, falls back to **OpenRouter** (cloud) with model rotation
- Generates structured insights: overview, key metrics, insights, anomalies, recommendations
- Parses JSON from the LLM response (handles markdown fences, partial JSON, etc.)

### Step 8: Visualization
```
All Reports + DataFrame → plots.py → 16+ Interactive Charts
```
- Every chart includes computed annotations (mean lines, correlation values, counts)
- Descriptive subtitles with actual statistics (not guesses)
- Color-coded legends explaining what each element represents

### Step 9: AI Consultant (On-Demand)
```
User Question + Context → ai_agent.py → AI Response
```
- Full conversation history is maintained
- AI has access to all analysis reports
- Can suggest Pandas code, chart types, and statistical tests

---

## 📦 Modules Explained (In Simple Terms)

### `app.py` — The Main Application
**What it does**: This is the "brain" that connects everything together. It creates the web pages, handles file uploads, and shows results.

**Simple analogy**: Think of it as the **front desk receptionist** — it takes your file, sends it to the right department, and presents the results back to you.

### `utils/data_handler.py` — Data Loader & Cleaner
**What it does**: Opens your CSV/Excel file and cleans it up (fixes messy column names, removes empty rows, optimizes data types).

**Simple analogy**: Like a **mail room** that opens your package, removes the bubble wrap, and organizes the contents neatly.

### `utils/schema_intel.py` — Schema Intelligence
**What it does**: Figures out what each column in your data *means* — is it a number? A category? A date? Could it be a prediction target?

**Simple analogy**: Like a **librarian** who looks at a new book and categorizes it by genre, language, and topic.

### `utils/data_quality.py` — Data Quality Auditor
**What it does**: Checks your data for problems — missing values, duplicates, outliers, and formatting issues. Gives a grade (A/B/C/D/F).

**Simple analogy**: Like a **quality inspector** at a factory checking products for defects before they ship.

### `utils/stats_profiler.py` — Statistical Profiler
**What it does**: Runs deep statistical analysis — distributions, correlations, which features are most important, and natural groupings in the data.

**Simple analogy**: Like a **detective** who looks for patterns, connections, and hidden groups in the evidence.

### `utils/plots.py` — Visualization Engine
**What it does**: Creates 16+ interactive charts automatically based on your data. Each chart has accurate annotations and legends.

**Simple analogy**: Like a **graphic designer** who turns raw numbers into beautiful, informative pictures.

### `utils/ai_agent.py` — AI Brain
**What it does**: Connects to an LLM (Ollama locally or OpenRouter cloud) to generate human-readable insights and answer your data questions.

**Simple analogy**: Like hiring a **senior data analyst** who reads all the reports and writes an executive summary for you.

### `utils/memory.py` — Session Memory
**What it does**: Remembers everything about your current analysis session — data, reports, chat history — so you don't lose context.

**Simple analogy**: Like a **notebook** that keeps all your meeting notes organized and accessible.

### `config.py` — App Configuration & Styling
**What it does**: Sets up the page layout and injects the premium dark theme CSS.

**Simple analogy**: Like an **interior designer** who sets up the office before you walk in.

---

## 🔬 Models & Algorithms Used

### 1. IQR Outlier Detection
- **What**: Interquartile Range method for finding outliers
- **How**: Calculate Q1 (25th percentile) and Q3 (75th percentile). IQR = Q3 - Q1. Any value below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is an outlier.
- **Why**: Simple, robust, works without assuming data follows a normal distribution.
- **Used in**: `data_quality.py → outlier_detection()`

### 2. D'Agostino-Pearson Normality Test
- **What**: A statistical test that checks if data follows a bell curve (normal distribution)
- **How**: Combines skewness and kurtosis tests into a single measure. A p-value < 0.05 means data is NOT normal.
- **Why**: Knowing if data is normal helps choose the right statistical methods.
- **Used in**: `stats_profiler.py → distribution_analysis()`

### 3. Pearson Correlation
- **What**: Measures the linear relationship between two numeric variables
- **How**: Returns a value from -1 (perfect negative) to +1 (perfect positive). 0 means no linear relationship.
- **Why**: Quickly identifies which features move together (useful for feature selection and understanding data).
- **Used in**: `stats_profiler.py → correlation_analysis()`

### 4. Mutual Information (Feature Importance)
- **What**: Measures how much knowing one variable tells you about another
- **How**: Uses information theory — high mutual information = strong dependency (works with non-linear relationships too)
- **Why**: Better than correlation alone because it catches non-linear patterns.
- **Used in**: `stats_profiler.py → _mutual_info_importance()` (via scikit-learn)

### 5. K-Means Clustering
- **What**: Groups similar data points into K clusters
- **How**: 
  1. Pick K random centers
  2. Assign each point to the nearest center
  3. Move centers to the average of their assigned points
  4. Repeat until stable
- **Why**: Discovers natural groupings without needing labels.
- **Used in**: `stats_profiler.py → basic_clustering()`

### 6. Silhouette Score (Optimal K Selection)
- **What**: Measures how well each data point fits its assigned cluster
- **How**: For each point, compare its distance to its own cluster vs the nearest other cluster. Score ranges from -1 (wrong cluster) to +1 (perfect fit).
- **Why**: Automatically picks the best number of clusters instead of guessing.
- **Used in**: `stats_profiler.py → basic_clustering()`

### 7. LLM (Large Language Model)  
- **Primary — Ollama (llama3.2:3b)**: Runs locally on your machine. 3 billion parameters. No internet needed. No rate limits. Free.
- **Fallback — OpenRouter**: Cloud-based. Rotates through 4 free models if one is rate-limited:
  - `google/gemma-3-27b-it:free`
  - `meta-llama/llama-3.3-70b-instruct:free`
  - `mistralai/mistral-small-3.1-24b-instruct:free`
  - `qwen/qwen3-4b:free`
- **Used for**: Executive summary generation and conversational data consulting.

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Ollama** (recommended for local LLM) — [Install Ollama](https://ollama.com)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama (Recommended)
```bash
# Install the local model (2GB download, runs free with no limits)
ollama pull llama3.2:3b

# Start the Ollama server (if not already running)
ollama serve
```

### 4. Set Up OpenRouter (Optional Fallback)
If you want cloud LLM as a fallback:
1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Create a free API key
3. Create a `.env` file:
```env
OPENROUTER_API_KEY=sk-or-v1-your_key_here
```

### 5. Run the App
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📖 Usage

1. **Upload** a CSV or Excel file using the sidebar
2. The app **automatically runs** the full analysis pipeline (takes ~10-30 seconds)
3. Explore 5 dashboard tabs:
   - **Executive Summary** — AI insights, anomalies, recommendations
   - **Schema Intelligence** — Column types, targets, primary keys
   - **Data Quality** — Missing values, outliers, duplicates
   - **Statistical Profile** — Distributions, correlations, feature importance
   - **Visualizations** — 16+ interactive charts
4. Switch to **AI Consultant** to ask questions about your data in plain English

---

## 📂 Project Structure

```
ai-data-analyst/
├── app.py                    # Main Streamlit application
├── config.py                 # Page setup + premium dark CSS theme
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed to git)
│
└── utils/
    ├── __init__.py
    ├── data_handler.py       # CSV/Excel loading + cleaning
    ├── schema_intel.py       # Schema intelligence engine
    ├── data_quality.py       # Data quality analysis
    ├── stats_profiler.py     # Statistical profiling
    ├── plots.py              # Plotly visualization engine
    ├── ai_agent.py           # LLM integration (Ollama + OpenRouter)
    └── memory.py             # Session state memory manager
```

---

## ⚙️ Configuration

### Changing the LLM Model

Edit `utils/ai_agent.py`:

```python
# Local (Ollama) — change to any model you've pulled
OLLAMA_MODEL = "llama3.2:3b"  # options: llama3, mistral, gemma2, etc.

# Cloud (OpenRouter) — reorder or swap models
OPENROUTER_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    # Add any openrouter.ai/models here
]
```

### Supported File Formats
- `.csv` — Comma-separated values (UTF-8 or Latin-1 encoding)
- `.xlsx` — Modern Excel spreadsheets
- `.xls` — Legacy Excel spreadsheets

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using Streamlit, Plotly, and AI**
