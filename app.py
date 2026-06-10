from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Student Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


RISK_LABELS = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
DATA_PATH = Path(__file__).parent / "data" / "dataset_student.xlsx"
CATEGORICAL_COLUMNS = [
    "Kondisi_Finansial (pendapatan ortu/bulan)",
    "Aktivitas_Organisasi(berapa banyak kegiatan)",
    "Kategori_MK_Sulit",
    "Status_Bekerja",
    "Tingkat_Stres_Akademik (jam tidur)",
]
MODEL_FEATURES = [
    "IPK_S1",
    "IPK_S2",
    "IPK_S3",
    "Total_SKS_Lulus",
    "Jumlah_MK_Ulang",
    "Eligible_Remidial (berdasarkan absen)",
    "Kondisi_Finansial (pendapatan ortu/bulan)",
    "Aktivitas_Organisasi(berapa banyak kegiatan)",
    "Jam_Kegiatan_Mingguan",
    "Kategori_MK_Sulit",
    "Status_Bekerja",
    "Jarak_Tempat_Tinggal_km",
    "Tingkat_Stres_Akademik (jam tidur)",
]
NUMERIC_COLUMNS = [col for col in MODEL_FEATURES if col not in CATEGORICAL_COLUMNS]
REQUIRED_COLUMNS = ["IPK_S4", *MODEL_FEATURES]
RISK_COLORS = {"Rendah": "#14704F", "Sedang": "#FFA934", "Tinggi": "#ED0086"}
ACCENT_COLORS = ["#F478B0", "#FFEF5A", "#ED0086", "#FFA934", "#00A0B5", "#98C54E", "#D6CFF8", "#F25823"]
FLOW_ITEMS = [
    ("Overview", "Home"),
    ("EDA", "Explore"),
    ("Models", "Score"),
    ("Features", "Rank"),
    ("Simulator", "Predict"),
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
        html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stSelectbox, .stNumberInput, .stMultiSelect {
            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
        }
        p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
            color: #1A1A1A;
        }
	        h1, h2, h3, h4, h5, h6 {
	            color: #1A1A1A;
	            font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
	            font-weight: 700;
	        }
	        /* Header tidak disembunyikan total supaya tombol buka/tutup sidebar tetap hidup. */
	        #MainMenu,
	        footer, 
	        [data-testid="stDecoration"] {
            display: none !important;
            }
	        [data-testid="stDecoration"] {
	            display: none !important;
	        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E7BEF8;
        }
        [data-testid="stSidebar"] * {
            color: #1A1A1A;
        }
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="tag"],
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #1A1A1A !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #E7BEF8 !important;
            border-color: #93ABD9 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #1A1A1A !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #F2619C !important;
            border: 0 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #EDE986;
            border-color: #93ABD9;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #1A1A1A;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: #93ABD9;
            border: 1px solid #6D85B3;
            color: #1A1A1A;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] {
            accent-color: #F2619C;
        }
        [data-testid="stSidebar"] [role="slider"] {
            background: #F2619C !important;
            border-color: #F2619C !important;
        }
	        [data-baseweb="tab-list"] {
	            gap: 0.5rem;
	            padding-top: 0 !important;
	            margin-top: 0 !important;
	        }
	        [data-baseweb="tab"] {
	            color: #4B403C;
	            font-weight: 700;
	            padding-top: 0.35rem;
	            padding-bottom: 0.35rem;
	        }
        [aria-selected="true"] {
            color: #ED0086 !important;
        }
        .stButton > button {
            background: #1A1A1A;
            border: 0;
            color: #FFF9E8;
            font-weight: 700;
        }
        .stButton > button:hover {
            background: #ED0086;
            color: #FFF9E8;
        }
        .stApp {
            background: #FFF9E8;
            color: #1A1A1A;
        }
	        .block-container {
	            max-width: 100%;
	            padding: 3.0rem 1.1rem 0.8rem 1.1rem;
	        }
	        [data-testid="stVerticalBlock"] {
	            gap: 0.45rem;
	        }
	        [data-testid="stHorizontalBlock"] {
	            gap: 0.7rem;
	        }
	        [data-testid="stWidgetLabel"] p {
	            font-size: 0.78rem;
	            line-height: 1.1;
	            margin-bottom: 0.1rem;
	        }
	        [data-testid="stNumberInput"] input,
	        [data-baseweb="select"] > div {
	            min-height: 2.05rem !important;
	        }
	        [data-testid="stNumberInput"] input {
	            padding-top: 0.2rem;
	            padding-bottom: 0.2rem;
	        }
	        [data-testid="stExpander"] {
	            border-color: #F4E4B1;
	        }
	        .hero-card {
	            background: #F478B0;
	            border-radius: 22px;
	            padding: 1.05rem 1.35rem;
	            color: #1A1A1A;
	            box-shadow: 0 12px 24px rgba(237, 0, 134, 0.14);
	        }
	        .hero-title {
	            font-size: 2.65rem;
	            font-weight: 700;
	            margin: 0 0 0.2rem 0;
	            line-height: 1;
	        }
	        .hero-subtitle {
	            font-size: 0.95rem;
	            line-height: 1.35;
	            margin: 0;
	            color: #1A1A1A;
	        }
	        .metric-card {
	            background: #FFFFFF;
	            border: 1px solid #F4E4B1;
	            border-radius: 16px;
	            padding: 0.72rem 0.85rem;
	            box-shadow: 0 8px 18px rgba(26, 26, 26, 0.07);
	        }
	        .metric-label {
	            font-size: 0.76rem;
	            color: #5C4A43;
	            margin-bottom: 0.18rem;
	        }
	        .metric-value {
	            font-size: 1.42rem;
	            font-weight: 700;
	            color: #1A1A1A;
	        }
        .insight-chip {
            display: inline-block;
            margin: 0.2rem 0.35rem 0 0;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #FFEF5A;
            color: #1A1A1A;
            font-size: 0.85rem;
            font-weight: 600;
        }
	        .section-note {
	            color: #5C4A43;
	            line-height: 1.35;
	            font-size: 0.86rem;
	        }
	        .flow-strip {
	            display: flex;
	            flex-wrap: wrap;
	            gap: 0.5rem;
	            margin: 0.35rem 0 0.95rem 0;
	        }
	        .flow-pill {
	            border: 1px solid #F4E4B1;
	            border-radius: 999px;
	            padding: 0.25rem 0.55rem;
	            background: #FFFFFF;
	            color: #4B403C;
	            font-size: 0.76rem;
	            font-weight: 600;
	        }
        .flow-pill.active {
            border-color: #F2619C;
            background: #F2619C;
            color: #FFFFFF;
        }
	        .field-card-title {
	            color: #1A1A1A;
	            font-size: 0.88rem;
	            font-weight: 700;
	            margin-bottom: 0.25rem;
	        }
	        .mini-note {
	            background: #FFFFFF;
	            border: 1px solid #F4E4B1;
	            border-radius: 14px;
	            padding: 0.65rem 0.8rem;
	            color: #4B403C;
	            line-height: 1.32;
	            font-size: 0.86rem;
	        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def label_risiko(ipk: float) -> int:
    if ipk < 2.75:
        return 2
    if ipk <= 3.25:
        return 1
    return 0


def map_income_to_category(income_rupiah: int) -> str:
    if income_rupiah < 3_000_000:
        return "Rendah"
    if income_rupiah <= 7_000_000:
        return "Menengah"
    return "Tinggi"


def format_rupiah(amount: int) -> str:
    return f"Rp {amount:,.0f}".replace(",", ".")


@st.cache_data
def load_dataset(uploaded_file: bytes | None = None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_excel(BytesIO(uploaded_file))
    return pd.read_excel(DATA_PATH)


def validate_and_clean_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = dataset.copy()
    messages: list[str] = []

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    if "NIM" not in df.columns:
        df.insert(0, "NIM", [f"MHS-{idx + 1:03d}" for idx in range(len(df))])
        messages.append("NIM column was not found, so temporary student IDs were generated.")
    elif df["NIM"].isna().any():
        missing_nim = int(df["NIM"].isna().sum())
        generated_ids = [f"MHS-{idx + 1:03d}" for idx in range(len(df))]
        df["NIM"] = df["NIM"].fillna(pd.Series(generated_ids, index=df.index))
        messages.append(f"{missing_nim} missing NIM value(s) were replaced with temporary IDs.")

    df["IPK_S4"] = pd.to_numeric(df["IPK_S4"], errors="coerce")
    missing_target = int(df["IPK_S4"].isna().sum())
    if missing_target:
        df = df.dropna(subset=["IPK_S4"]).copy()
        messages.append(f"{missing_target} row(s) with missing IPK_S4 were removed because the target label needs that value.")

    if df.empty:
        raise ValueError("Dataset has no usable rows after cleaning missing IPK_S4 values.")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_count = int(df[col].isna().sum())
        if missing_count:
            median_value = df[col].median()
            fill_value = 0 if pd.isna(median_value) else median_value
            df[col] = df[col].fillna(fill_value)
            messages.append(f"{missing_count} missing value(s) in {col} were filled with {fill_value:.2f}.")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
        missing_count = int(df[col].isna().sum())
        if missing_count:
            mode_values = df[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Tidak diketahui"
            df[col] = df[col].fillna(fill_value)
            messages.append(f"{missing_count} missing value(s) in {col} were filled with '{fill_value}'.")

    return df, messages


def calculate_woe_iv(dataset: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    rows = []
    for value in dataset[feature].dropna().unique():
        subset = dataset[dataset[feature] == value]
        rows.append(
            {
                "Value": value,
                "All": subset[feature].count(),
                "Good": subset[subset[target] == 0][feature].count(),
                "Bad": subset[subset[target] != 0][feature].count(),
            }
        )

    result = pd.DataFrame(rows)
    result["Dist_Good"] = result["Good"] / result["Good"].sum()
    result["Dist_Bad"] = result["Bad"] / result["Bad"].sum()
    result["WoE"] = np.log((result["Dist_Good"] + 0.001) / (result["Dist_Bad"] + 0.001))
    result["IV"] = (result["Dist_Good"] - result["Dist_Bad"]) * result["WoE"]
    return result.replace([np.inf, -np.inf], 0)


def build_iv_table(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "IPK_S1",
        "IPK_S2",
        "IPK_S3",
        "Jumlah_MK_Ulang",
        "Kondisi_Finansial (pendapatan ortu/bulan)",
        "Aktivitas_Organisasi(berapa banyak kegiatan)",
        "Status_Bekerja",
        "Tingkat_Stres_Akademik (jam tidur)",
    ]
    work_df = df.copy()
    iv_results = []

    for col in features:
        if "IPK" in col:
            binned = pd.qcut(work_df[col], q=4, duplicates="drop")
            temp_col = f"{col}_bins"
            work_df[temp_col] = binned.astype(str)
        else:
            temp_col = col

        iv_value = calculate_woe_iv(work_df, temp_col, "Target_Risiko")["IV"].sum()

        if iv_value < 0.02:
            strength = "Useless"
        elif iv_value < 0.1:
            strength = "Weak"
        elif iv_value < 0.3:
            strength = "Medium"
        elif iv_value < 0.5:
            strength = "Strong"
        else:
            strength = "Suspicious"

        iv_results.append({"Fitur": col, "IV_Score": iv_value, "Strength": strength})

    return pd.DataFrame(iv_results).sort_values("IV_Score", ascending=False).reset_index(drop=True)


@st.cache_resource
def train_pipeline(uploaded_file: bytes | None = None) -> dict:
    df_loaded = load_dataset(uploaded_file).copy()
    df_raw, preprocessing_messages = validate_and_clean_dataset(df_loaded)
    df_raw["Target_Risiko"] = df_raw["IPK_S4"].apply(label_risiko)
    df_encoded = df_raw.copy()

    encoders: dict[str, LabelEncoder] = {}
    category_options: dict[str, list[str]] = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoders[col] = encoder
        category_options[col] = list(encoder.classes_)

    X = df_encoded[MODEL_FEATURES]
    y = df_encoded["Target_Risiko"]

    if y.nunique() < 2:
        raise ValueError("Dataset needs at least two risk categories after preprocessing to train the model.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        ),
    }

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    ensemble_model = VotingClassifier(
        estimators=[
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
            ),
        ],
        voting="soft",
    )
    ensemble_model.fit(X_train, y_train)
    predictions["Ensemble (Soft Voting)"] = ensemble_model.predict(X_test)
    probabilities["Ensemble (Soft Voting)"] = ensemble_model.predict_proba(X_test)

    metrics_df = (
        pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, pred),
                    "F1-Score": f1_score(y_test, pred, average="weighted"),
                }
                for name, pred in predictions.items()
            ]
        )
        .sort_values(["Accuracy", "F1-Score"], ascending=False)
        .reset_index(drop=True)
    )

    confusion = confusion_matrix(y_test, predictions["Ensemble (Soft Voting)"])
    feature_importance_df = (
        pd.DataFrame(
            {
                "Feature": MODEL_FEATURES,
                "Importance": models["XGBoost"].feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    iv_df = build_iv_table(df_raw)

    return {
        "df_raw": df_raw,
        "df_encoded": df_encoded,
        "encoders": encoders,
        "category_options": category_options,
        "scaler": scaler,
        "models": models,
        "ensemble_model": ensemble_model,
        "metrics_df": metrics_df,
        "confusion": confusion,
        "feature_importance_df": feature_importance_df,
        "iv_df": iv_df,
        "preprocessing_messages": preprocessing_messages,
    }


def create_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow(active_index: int) -> None:
    pills = []
    for index, (title, label) in enumerate(FLOW_ITEMS):
        active_class = " active" if index == active_index else ""
        pills.append(f'<span class="flow-pill{active_class}">{index + 1}. {label}</span>')
    st.markdown(f'<div class="flow-strip">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_usage_popover() -> None:
    with st.sidebar.popover("How to Use"):
        st.markdown(
            """
            **Quick guide**

            1. Upload a dataset only if you want to replace the default file.
            2. Use the filters to narrow the visible student population.
            3. Open each tab from left to right: overview, EDA, model, features, simulator.
            4. In the simulator, adjust the student profile and read the risk output.
            """
        )


def show_status_toast(preprocessing_messages: list[str], uploaded_file: bytes | None) -> None:
    data_key = f"{len(uploaded_file) if uploaded_file else 'default'}:{len(preprocessing_messages)}"
    if st.session_state.get("last_status_toast") == data_key:
        return

    if preprocessing_messages:
        st.toast("Preprocessing completed. Missing values were handled automatically.", icon="✅")
    else:
        st.toast("Dashboard ready. Dataset loaded successfully.", icon="✅")
    st.session_state["last_status_toast"] = data_key


def render_preprocessing_notes(messages: list[str]) -> None:
    if not messages:
        return

    with st.sidebar.popover("Data Notes"):
        st.markdown("**Preprocessing messages**")
        for message in messages:
            st.markdown(f"- {message}")


def style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font={"family": "Space Grotesk, Avenir Next, Segoe UI, sans-serif", "color": "#1A1A1A"},
        title_font={"color": "#1A1A1A", "size": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"gridcolor": "#F4E4B1", "linecolor": "#F4E4B1", "zerolinecolor": "#F4E4B1"},
        yaxis={"gridcolor": "#F4E4B1", "linecolor": "#F4E4B1", "zerolinecolor": "#F4E4B1"},
    )
    return fig


def render_overview(df: pd.DataFrame, metrics_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    top_model = metrics_df.iloc[0]
    risk_counts = filtered_df["Target_Risiko"].map(RISK_LABELS).value_counts()
    risk_order_for_text = ["Sedang", "Rendah", "Tinggi"]
    risk_summary = ", ".join(
        f"{label} {int(risk_counts.get(label, 0))}" for label in risk_order_for_text
    )

    render_flow(0)

    # Header dibuat full-width supaya tidak tabrakan dengan card metrik di bawahnya.
    st.markdown(
        """
        <div class="hero-card" style="margin-bottom:1.15rem;">
            <div class="hero-title">Student Risk Dashboard</div>
            <p class="hero-subtitle">
                Ringkasan risiko akademik, performa model, dan simulasi prediksi dalam satu tampilan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Card ringkasan dipisahkan ke baris sendiri agar layout tidak menumpuk.
    summary_cols = st.columns([1.2, 0.9, 0.9, 0.9, 0.9], gap="large")

    with summary_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height:108px;">
                <div class="metric-label">Best Model Saat Ini</div>
                <div class="metric-value">{top_model["Model"]}</div>
                <div class="section-note">Accuracy {top_model["Accuracy"]:.2%} • F1 {top_model["F1-Score"]:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_cols[1]:
        create_metric_card("Total Mahasiswa", f"{len(filtered_df)}")

    with summary_cols[2]:
        create_metric_card("Rata-rata IPK S4", f"{filtered_df['IPK_S4'].mean():.2f}")

    with summary_cols[3]:
        create_metric_card("Rata-rata SKS Lulus", f"{filtered_df['Total_SKS_Lulus'].mean():.1f}")

    with summary_cols[4]:
        create_metric_card("MK Ulang Rata-rata", f"{filtered_df['Jumlah_MK_Ulang'].mean():.1f}")

    st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

    top_feature = "IPK_S3"
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:0.55rem; margin:0.2rem 0 1.65rem 0;">
            <span class="insight-chip">Distribusi Risiko: {risk_summary}</span>
            <span class="insight-chip">Faktor Dominan: {top_feature}</span>
            <span class="insight-chip">Dataset: {df.shape[0]} baris, {df.shape[1]} kolom</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_left, overview_right = st.columns([1.05, 1], gap="large")

    with overview_left:
        risk_order = ["Rendah", "Sedang", "Tinggi"]
        risk_chart_df = (
            filtered_df["Target_Risiko"]
            .map(RISK_LABELS)
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )
        risk_chart_df.columns = ["Risiko", "Jumlah"]

        fig = px.bar(
            risk_chart_df,
            x="Risiko",
            y="Jumlah",
            color="Risiko",
            color_discrete_map=RISK_COLORS,
            text_auto=True,
        )
        fig.update_layout(
            title="Distribusi Kategori Risiko",
            height=315,
            margin=dict(l=20, r=20, t=60, b=35),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with overview_right:
        ipk_trend_df = (
            filtered_df[["IPK_S1", "IPK_S2", "IPK_S3", "IPK_S4"]]
            .mean()
            .reset_index()
        )

        fig = px.line(
            ipk_trend_df,
            x="index",
            y=0,
            markers=True,
        )
        fig.update_traces(
            line_color="#ED0086",
            marker_color="#FFEF5A",
            line_width=4,
            marker_size=10,
        )
        fig.update_layout(
            title="Rata-rata Tren IPK per Semester",
            xaxis_title="Semester",
            yaxis_title="IPK Rata-rata",
            height=315,
            margin=dict(l=20, r=20, t=60, b=35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            style_chart(fig),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    with st.expander("Preview filtered rows", expanded=False):
        st.dataframe(filtered_df.head(20), use_container_width=True, height=220)


def render_eda(filtered_df: pd.DataFrame) -> None:
    render_flow(1)
    st.markdown("### Exploratory Data Analysis")
    selected_visual = st.radio(
        "EDA visual",
        ["GPA", "Retake vs GPA", "Correlation", "Stress"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_visual == "GPA":
        fig = px.box(
            filtered_df.melt(
                value_vars=["IPK_S1", "IPK_S2", "IPK_S3", "IPK_S4"],
                var_name="Semester",
                value_name="IPK",
            ),
            x="Semester",
            y="IPK",
            color="Semester",
            color_discrete_sequence=ACCENT_COLORS,
        )
        fig.update_layout(
            title="Distribusi IPK Mahasiswa per Semester",
            height=365,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    elif selected_visual == "Retake vs GPA":
        fig = px.scatter(
            filtered_df,
            x="Jumlah_MK_Ulang",
            y="IPK_S4",
            color=filtered_df["Target_Risiko"].map(RISK_LABELS),
            color_discrete_map=RISK_COLORS,
            hover_data=["NIM", "Total_SKS_Lulus", "Jarak_Tempat_Tinggal_km"],
        )
        fig.update_layout(
            title="Jumlah MK Ulang vs IPK Semester 4",
            height=365,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    elif selected_visual == "Correlation":
        numeric_df = filtered_df.select_dtypes(include=[np.number]).drop(columns=["NIM"], errors="ignore")
        corr = numeric_df.corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale=[
                    [0, "#F25823"],
                    [0.5, "#FFF9E8"],
                    [1, "#00A0B5"],
                ],
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Heatmap Korelasi",
            height=365,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        stress_order = sorted(filtered_df["Tingkat_Stres_Akademik (jam tidur)"].unique().tolist())
        fig = px.histogram(
            filtered_df,
            x="Tingkat_Stres_Akademik (jam tidur)",
            color="Status_Bekerja",
            category_orders={"Tingkat_Stres_Akademik (jam tidur)": stress_order},
            color_discrete_sequence=["#F478B0", "#FFA934", "#00A0B5"],
            barmode="group",
        )
        fig.update_layout(
            title="Distribusi Tingkat Stres Akademik",
            height=365,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

    st.plotly_chart(style_chart(fig), use_container_width=True)


def render_model_evaluation(metrics_df: pd.DataFrame, confusion: np.ndarray) -> None:
    render_flow(2)
    st.markdown("### Evaluasi Model")

    col1, col2 = st.columns([1, 1])
    with col1:
        metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(
            metrics_long,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#ED0086", "#FFA934"],
            text_auto=".3f",
        )
        fig.update_layout(
            title="Perbandingan Accuracy dan F1-Score",
            height=315,
            yaxis_range=[0, 1.05],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
        with st.expander("Model score table", expanded=False):
            st.dataframe(
                metrics_df.style.format({"Accuracy": "{:.2%}", "F1-Score": "{:.2%}"}),
                use_container_width=True,
                height=180,
            )

    with col2:
        fig = go.Figure(
            data=go.Heatmap(
                z=confusion,
                x=["Rendah", "Sedang", "Tinggi"],
                y=["Rendah", "Sedang", "Tinggi"],
                colorscale=[[0, "#D6CFF8"], [0.5, "#FFEF5A"], [1, "#00A0B5"]],
                text=confusion,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Confusion Matrix Ensemble (Soft Voting)",
            xaxis_title="Prediksi",
            yaxis_title="Aktual",
            height=315,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.markdown('<div class="mini-note">Matrix ini membandingkan prediksi model dengan label aktual pada data uji.</div>', unsafe_allow_html=True)


def render_feature_analysis(feature_importance_df: pd.DataFrame, iv_df: pd.DataFrame) -> None:
    render_flow(3)
    st.markdown("### Feature Analysis")

    selected_visual = st.radio(
        "Feature visual",
        ["XGBoost Importance", "Information Value"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_visual == "XGBoost Importance":
        fig = px.bar(
            feature_importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#FFEF5A", "#ED0086"],
        )
        fig.update_layout(
            title="Top Feature Importance dari XGBoost",
            height=380,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig = px.bar(
            iv_df,
            x="IV_Score",
            y="Fitur",
            color="Strength",
            orientation="h",
            color_discrete_map={
                "Useless": "#D6CFF8",
                "Weak": "#FFEF5A",
                "Medium": "#98C54E",
                "Strong": "#00A0B5",
                "Suspicious": "#F25823",
            },
        )
        fig.update_layout(
            title="Information Value (IV) per Fitur",
            height=380,
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

    st.plotly_chart(style_chart(fig), use_container_width=True)
    with st.expander("Top factors table", expanded=False):
        st.dataframe(feature_importance_df.head(10), use_container_width=True, height=220)


def render_prediction_form(pipeline: dict) -> None:
    render_flow(4)
    st.markdown("### Risk Simulator")

    df_raw = pipeline["df_raw"]
    category_options = pipeline["category_options"]
    scaler = pipeline["scaler"]
    model = pipeline["ensemble_model"]

    input_col, result_col = st.columns([2.35, 0.95])
    with input_col:
        academic_col, activity_col, context_col = st.columns(3)
        with academic_col:
            with st.container(border=True):
                st.markdown('<div class="field-card-title">Academic Profile</div>', unsafe_allow_html=True)
                ipk_s1 = st.number_input("IPK Semester 1", min_value=0.0, max_value=4.0, value=float(df_raw["IPK_S1"].median()), step=0.01)
                ipk_s2 = st.number_input("IPK Semester 2", min_value=0.0, max_value=4.0, value=float(df_raw["IPK_S2"].median()), step=0.01)
                ipk_s3 = st.number_input("IPK Semester 3", min_value=0.0, max_value=4.0, value=float(df_raw["IPK_S3"].median()), step=0.01)
                total_sks = st.number_input("Total SKS Lulus", min_value=0, max_value=160, value=int(df_raw["Total_SKS_Lulus"].median()), step=1)
                mk_ulang = st.number_input("Jumlah MK Ulang", min_value=0, max_value=20, value=int(df_raw["Jumlah_MK_Ulang"].median()), step=1)

        with activity_col:
            with st.container(border=True):
                st.markdown('<div class="field-card-title">Finance & Activity</div>', unsafe_allow_html=True)
                pendapatan = st.number_input(
                    "Pendapatan Ortu / Bulan (Rp)",
                    min_value=0,
                    max_value=100_000_000,
                    value=5_000_000,
                    step=500_000,
                )
                derived_finansial = map_income_to_category(pendapatan)
                st.caption(f"{derived_finansial} | {format_rupiah(pendapatan)}")
                organisasi = st.selectbox("Aktivitas Organisasi", options=category_options["Aktivitas_Organisasi(berapa banyak kegiatan)"])
                jam_kegiatan = st.number_input("Jam Kegiatan Mingguan", min_value=0, max_value=80, value=int(df_raw["Jam_Kegiatan_Mingguan"].median()), step=1)
                status_bekerja = st.selectbox("Status Bekerja", options=category_options["Status_Bekerja"])

        with context_col:
            with st.container(border=True):
                st.markdown('<div class="field-card-title">Context & Wellbeing</div>', unsafe_allow_html=True)
                eligible = st.selectbox(
                    "Eligible Remidial",
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                )
                mk_sulit = st.selectbox("Kategori MK Sulit", options=category_options["Kategori_MK_Sulit"])
                jarak = st.number_input("Jarak Tinggal (km)", min_value=0, max_value=300, value=int(df_raw["Jarak_Tempat_Tinggal_km"].median()), step=1)
                stres = st.selectbox("Tingkat Stres", options=category_options["Tingkat_Stres_Akademik (jam tidur)"])

    input_df = pd.DataFrame(
        [
            {
                "IPK_S1": ipk_s1,
                "IPK_S2": ipk_s2,
                "IPK_S3": ipk_s3,
                "Total_SKS_Lulus": total_sks,
                "Jumlah_MK_Ulang": mk_ulang,
                "Eligible_Remidial (berdasarkan absen)": eligible,
                "Kondisi_Finansial (pendapatan ortu/bulan)": pipeline["encoders"]["Kondisi_Finansial (pendapatan ortu/bulan)"].transform([derived_finansial])[0],
                "Aktivitas_Organisasi(berapa banyak kegiatan)": pipeline["encoders"]["Aktivitas_Organisasi(berapa banyak kegiatan)"].transform([organisasi])[0],
                "Jam_Kegiatan_Mingguan": jam_kegiatan,
                "Kategori_MK_Sulit": pipeline["encoders"]["Kategori_MK_Sulit"].transform([mk_sulit])[0],
                "Status_Bekerja": pipeline["encoders"]["Status_Bekerja"].transform([status_bekerja])[0],
                "Jarak_Tempat_Tinggal_km": jarak,
                "Tingkat_Stres_Akademik (jam tidur)": pipeline["encoders"]["Tingkat_Stres_Akademik (jam tidur)"].transform([stres])[0],
            }
        ]
    )
    scaled_input = scaler.transform(input_df[MODEL_FEATURES])
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    risk_label = RISK_LABELS[prediction]
    risk_color = RISK_COLORS[risk_label]

    with result_col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Hasil Prediksi</div>
                <div class="metric-value" style="color:{risk_color};">{risk_label}</div>
                <div class="section-note">Kategori risiko akademik berdasarkan profil input.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prob_df = pd.DataFrame(
            {
                "Risiko": [RISK_LABELS[i] for i in range(len(probabilities))],
                "Probabilitas": probabilities,
            }
        )
        fig = px.bar(
            prob_df,
            x="Risiko",
            y="Probabilitas",
            color="Risiko",
            color_discrete_map=RISK_COLORS,
            text_auto=".2%",
        )
        fig.update_layout(
            title="Probability",
            yaxis_tickformat=".0%",
            height=225,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

        recommendation = {
            "Rendah": "Pertahankan konsistensi akademik dan monitoring rutin.",
            "Sedang": "Evaluasi beban kuliah, stres, dan aktivitas mingguan.",
            "Tinggi": "Prioritaskan pendampingan akademik dan rencana perbaikan.",
        }
        st.markdown(f'<div class="mini-note"><b>Suggested action</b><br>{recommendation[risk_label]}</div>', unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    st.sidebar.title("Student Filters")
    render_usage_popover()

    uploaded_file = st.sidebar.file_uploader(
        "Upload Student Dataset",
        type=["xlsx", "xls"],
    )

    uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
    try:
        pipeline = train_pipeline(uploaded_bytes)
    except Exception as exc:
        st.error(f"Unable to prepare dashboard data: {exc}")
        return

    show_status_toast(pipeline["preprocessing_messages"], uploaded_bytes)
    render_preprocessing_notes(pipeline["preprocessing_messages"])

    df = pipeline["df_raw"].copy()
    df["Risk_Label"] = df["Target_Risiko"].map(RISK_LABELS)

    selected_risks = st.sidebar.multiselect(
        "Risk Category",
        options=["Rendah", "Sedang", "Tinggi"],
        default=["Rendah", "Sedang", "Tinggi"],
    )
    selected_status = st.sidebar.multiselect(
        "Employment Status",
        options=sorted(df["Status_Bekerja"].unique().tolist()),
        default=sorted(df["Status_Bekerja"].unique().tolist()),
    )
    ipk_range = st.sidebar.slider(
        "Semester 4 GPA Range",
        min_value=float(df["IPK_S4"].min()),
        max_value=float(df["IPK_S4"].max()),
        value=(float(df["IPK_S4"].min()), float(df["IPK_S4"].max())),
    )

    filtered_df = df[
        df["Risk_Label"].isin(selected_risks)
        & df["Status_Bekerja"].isin(selected_status)
        & df["IPK_S4"].between(ipk_range[0], ipk_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("Filter saat ini tidak menghasilkan data. Silakan longgarkan filter sidebar.")
        return

    tabs = st.tabs(
        [
            "🏠 Overview",
            "📊 EDA",
            "🤖 Models",
            "⭐ Features",
            "🎯 Simulator",
        ]
    )

    with tabs[0]:
        render_overview(df, pipeline["metrics_df"], filtered_df)
    with tabs[1]:
        render_eda(filtered_df)
    with tabs[2]:
        render_model_evaluation(pipeline["metrics_df"], pipeline["confusion"])
    with tabs[3]:
        render_feature_analysis(pipeline["feature_importance_df"], pipeline["iv_df"])
    with tabs[4]:
        render_prediction_form(pipeline)


if __name__ == "__main__":
    main()
