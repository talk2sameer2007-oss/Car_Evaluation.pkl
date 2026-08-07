import os
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr

# ==========================================================
# 1. Model Loading & Fallback Setup
# ==========================================================
MODEL_PATH = "Car_Evaluation.pkl"

try:
    deployed_xgb = joblib.load(MODEL_PATH)
    print("✅ XGBoost Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model load note: {e}. Running in pipeline simulation mode.")
    deployed_xgb = None

# Mappings for categorical display
LABEL_MAPS = {
    "buying": {0: "Low", 1: "Medium", 2: "High", 3: "Very High"},
    "maint": {0: "Low", 1: "Medium", 2: "High", 3: "Very High"},
    "doors": {2: "2", 3: "3", 4: "4", 5: "5+"},
    "persons": {2: "2", 4: "4", 5: "5+"},
    "lug_boot": {0: "Small", 1: "Medium", 2: "Big"},
    "safety": {0: "Low", 1: "Medium", 2: "High"}
}

RESULT_MAP = {
    0: ("Unacceptable (unacc)", "FAIL", "#ff4d4f"),
    1: ("Acceptable (acc)", "PASS", "#52c41a"),
    2: ("Good (good)", "PASS", "#1890ff"),
    3: ("Very Good (vgood)", "PASS", "#722ed1")
}

# Initial baseline dataset for the Live Log table & metrics
def get_initial_history():
    return pd.DataFrame([
        {"Eval_ID": "EV-1001", "Timestamp": "09:15:20", "Buying": "Medium", "Maint": "Low", "Doors": "4", "Persons": "4", "Boot": "Medium", "Safety": "High", "Decision": "Very Good (vgood)", "Status": "PASS"},
        {"Eval_ID": "EV-1002", "Timestamp": "09:32:10", "Buying": "High", "Maint": "High", "Doors": "2", "Persons": "2", "Boot": "Small", "Safety": "Low", "Decision": "Unacceptable (unacc)", "Status": "FAIL"},
        {"Eval_ID": "EV-1003", "Timestamp": "10:05:45", "Buying": "Low", "Maint": "Medium", "Doors": "4", "Persons": "4", "Boot": "Big", "Safety": "Medium", "Decision": "Acceptable (acc)", "Status": "PASS"},
        {"Eval_ID": "EV-1004", "Timestamp": "10:42:12", "Buying": "Medium", "Maint": "Medium", "Doors": "5+", "Persons": "5+", "Boot": "Big", "Safety": "High", "Decision": "Good (good)", "Status": "PASS"},
    ])


# ==========================================================
# 2. Analytics & Visual Generator Functions
# ==========================================================
def generate_trend_chart(df_history):
    """Generates the dark neon line graph matching the Revenue chart from the reference UI."""
    score_mapping = {"Unacceptable (unacc)": 1, "Acceptable (acc)": 2, "Good (good)": 3, "Very Good (vgood)": 4}
    y_values = [score_mapping.get(d, 1) for d in df_history["Decision"]]
    x_values = df_history["Eval_ID"].tolist()

    fig = go.Figure()

    # Dark gradient background glow line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name='Safety Level',
        line=dict(color='#00f2fe', width=3, shape='spline'),
        marker=dict(size=8, color='#ffffff', line=dict(color='#00f2fe', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.08)'
    ))

    fig.update_layout(
        title=dict(text="<b>Safety Telemetry Score Trend</b>", font=dict(color="#ffffff", size=15)),
        paper_bgcolor='#121621',
        plot_bgcolor='#121621',
        margin=dict(l=30, r=30, t=50, b=30),
        height=280,
        xaxis=dict(showgrid=False, zeroline=False, color='#7a889b', tickfont=dict(size=10)),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#7a889b',
            tickvals=[1, 2, 3, 4], ticktext=['Unacc', 'Acc', 'Good', 'VGood']
        ),
        showlegend=False
    )
    return fig


def generate_distribution_chart(df_history):
    """Generates class distribution breakdown."""
    counts = df_history["Decision"].value_counts().reset_index()
    counts.columns = ["Decision", "Count"]
    
    fig = px.pie(
        counts, 
        names="Decision", 
        values="Count", 
        color="Decision",
        color_discrete_map={
            "Unacceptable (unacc)": "#ff4d4f",
            "Acceptable (acc)": "#52c41a",
            "Good (good)": "#1890ff",
            "Very Good (vgood)": "#722ed1"
        },
        hole=0.65
    )
    
    fig.update_layout(
        title=dict(text="<b>Class Distribution</b>", font=dict(color="#222", size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, font=dict(size=10))
    )
    return fig


def create_kpi_card(title, value, subtitle, trend_color="#52c41a"):
    """Generates modern light KPI metric cards like the top row in reference UI."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub" style="color: {trend_color};">{subtitle}</div>
    </div>
    """


# ==========================================================
# 3. Core Prediction & Dashboard Controller
# ==========================================================
def process_evaluation(buying, maint, doors, persons, boot, safety, history_df):
    if any(v is None for v in [buying, maint, doors, persons, boot, safety]):
        err_card = """<div class="result-box alert-box">❌ Please complete all parameter selections on the console.</div>"""
        return history_df, history_df, generate_trend_chart(history_df), generate_distribution_chart(history_df), err_card, gr.update(), gr.update(), gr.update(), gr.update()

    # Predict using model or intelligent rule fallback
    feature_vec = [[int(buying), int(maint), int(doors), int(persons), int(boot), int(safety)]]
    
    if deployed_xgb is not None:
        try:
            pred_class = int(deployed_xgb.predict(feature_vec)[0])
        except Exception:
            pred_class = 0 if int(safety) == 0 else (1 if int(buying) >= 2 else 2)
    else:
        # Mock logic matching standard Car Evaluation heuristic if pkl missing
        if int(safety) == 0 or int(persons) == 2:
            pred_class = 0
        elif int(safety) == 2 and int(buying) <= 1 and int(maint) <= 1:
            pred_class = 3
        elif int(safety) >= 1 and int(buying) <= 2:
            pred_class = 2
        else:
            pred_class = 1

    decision_label, status, badge_color = RESULT_MAP.get(pred_class, ("Unacceptable", "FAIL", "#ff4d4f"))
    
    # New log entry
    new_id = f"EV-{1001 + len(history_df)}"
    now_str = datetime.now().strftime("%H:%M:%S")
    
    new_row = {
        "Eval_ID": new_id,
        "Timestamp": now_str,
        "Buying": LABEL_MAPS["buying"][int(buying)],
        "Maint": LABEL_MAPS["maint"][int(maint)],
        "Doors": LABEL_MAPS["doors"][int(doors)],
        "Persons": LABEL_MAPS["persons"][int(persons)],
        "Boot": LABEL_MAPS["lug_boot"][int(boot)],
        "Safety": LABEL_MAPS["safety"][int(safety)],
        "Decision": decision_label,
        "Status": status
    }
    
    # Append to state dataframe
    updated_df = pd.concat([pd.DataFrame([new_row]), history_df], ignore_index=True)
    
    # Calculate updated KPI stats
    total_evals = len(updated_df)
    pass_count = len(updated_df[updated_df["Status"] == "PASS"])
    pass_rate = f"{(pass_count / total_evals) * 100:.1f}%"
    high_safety_cnt = len(updated_df[updated_df["Safety"] == "High"])
    
    kpi1 = create_kpi_card("Total Evaluated", f"{total_evals} Vehicles", "↗ Live session total", "#1890ff")
    kpi2 = create_kpi_card("Acceptable Rate", pass_rate, f"↗ {pass_count} passed standards", "#52c41a")
    kpi3 = create_kpi_card("High Safety Grade", f"{high_safety_cnt} Units", "↗ Safety rating = High", "#722ed1")
    kpi4 = create_kpi_card("Latest Assessment", decision_label.split()[0], f"Status: {status}", badge_color)

    # Result card
    result_html = f"""
    <div class="result-box" style="border-left: 5px solid {badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; font-size: 0.85rem; color: #8c8c8c;">EVALUATION RESULT</span>
            <span class="status-badge" style="background: {badge_color}; color: #fff;">{status}</span>
        </div>
        <h2 style="margin: 8px 0; color: {badge_color}; font-size: 1.6rem; font-weight: 800;">{decision_label}</h2>
        <p style="margin: 0; color: #595959; font-size: 0.85rem;">Vehicle evaluated under ID <b>{new_id}</b> at {now_str}.</p>
    </div>
    """

    trend_fig = generate_trend_chart(updated_df)
    dist_fig = generate_distribution_chart(updated_df)

    return updated_df, updated_df, trend_fig, dist_fig, result_html, kpi1, kpi2, kpi3, kpi4


# ==========================================================
# 4. Modern Clean Dashboard CSS (Matching Reference UI)
# ==========================================================
DASHBOARD_CSS = """
:root {
    --bg-main: #f4f6fa;
    --card-bg: #ffffff;
    --text-primary: #1f2937;
    --border-color: #e5e7eb;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Header Navbar Styling */
.top-nav {
    background: #ffffff;
    border-bottom: 1px solid var(--border-color);
    padding: 16px 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

.top-nav-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
}

/* Top KPI Cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}

.kpi-title {
    font-size: 0.78rem;
    color: #6b7280;
    font-weight: 600;
    text-transform: uppercase;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #111827;
    margin: 6px 0;
}

.kpi-sub {
    font-size: 0.75rem;
    font-weight: 600;
}

/* Card Containers */
.dashboard-card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
}

/* Primary Action Button */
.btn-primary {
    background: #111827 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.2) !important;
}

.btn-primary:hover {
    background: #1f2937 !important;
}

/* Results & Badges */
.result-box {
    background: #f9fafb;
    border-radius: 12px;
    padding: 16px;
    margin-top: 15px;
    border: 1px solid var(--border-color);
}

.status-badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.alert-box {
    background: #fff2f0;
    color: #ff4d4f;
    border-color: #ffccc7;
    font-weight: 600;
}

/* Dataframe Clean Styling */
.gr-dataframe {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color) !important;
}
"""


# ==========================================================
# 5. Gradio Dashboard Interface Layout
# ==========================================================
with gr.Blocks(title="Car Safety & Evaluation System", css=DASHBOARD_CSS) as demo:
    
    # State storage for evaluation logs
    history_state = gr.State(get_initial_history())

    # Header Bar
    gr.HTML(
        """
        <div class="top-nav">
            <div class="top-nav-title">
                ⚡ CAR SAFETY AND EVALUATION PREDICTION SYSTEM
            </div>
            <div style="font-size: 0.85rem; color: #6b7280; font-weight: 600;">
                🟢 ML Telemetry Online &nbsp;|&nbsp; Model: XGBoost &nbsp;|&nbsp; Developer: Sameer
            </div>
        </div>
        """
    )

    # TOP KPI METRICS ROW (Matching Reference Dashboard Top Stat Cards)
    with gr.Row():
        kpi_1 = gr.HTML(create_kpi_card("Total Evaluated", "4 Vehicles", "↗ Live session total", "#1890ff"))
        kpi_2 = gr.HTML(create_kpi_card("Acceptable Rate", "75.0%", "↗ 3 passed standards", "#52c41a"))
        kpi_3 = gr.HTML(create_kpi_card("High Safety Grade", "2 Units", "↗ Safety rating = High", "#722ed1"))
        kpi_4 = gr.HTML(create_kpi_card("Latest Assessment", "Good", "Status: PASS", "#1890ff"))

    # MIDDLE ROW: Input Console + Live Trend Chart + Class Distribution
    with gr.Row():
        
        # COLUMN 1: Interactive Spec Selector & Run Diagnostic
        with gr.Column(scale=4, elem_classes=["dashboard-card"]):
            gr.Markdown("### 🎛️ **Vehicle Spec Inputs**")
            
            with gr.Row():
                buying_price = gr.Dropdown(
                    choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)],
                    label="Buying Price", value=1
                )
                maintenance_cost = gr.Dropdown(
                    choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)],
                    label="Maintenance Cost", value=1
                )

            with gr.Row():
                number_of_doors = gr.Dropdown(
                    choices=[("2", 2), ("3", 3), ("4", 4), ("5 or More", 5)],
                    label="Door Count", value=4
                )
                number_of_persons = gr.Dropdown(
                    choices=[("2", 2), ("4", 4), ("More (5+)", 5)],
                    label="Person Capacity", value=4
                )

            with gr.Row():
                lug_boot = gr.Dropdown(
                    choices=[("Small", 0), ("Medium", 1), ("Big", 2)],
                    label="Boot Size", value=1
                )
                safety = gr.Dropdown(
                    choices=[("Low", 0), ("Medium", 1), ("High", 2)],
                    label="Safety Level", value=2
                )

            btn_run = gr.Button("RUN EVALUATION DIAGNOSTIC ➔", elem_classes=["btn-primary"])

            # Instant Output Result Box
            result_output = gr.HTML(
                value="""
                <div class="result-box">
                    <span style="color: #8c8c8c; font-size: 0.85rem; font-weight: 600;">SYSTEM READY</span>
                    <p style="margin: 5px 0 0 0; color: #595959; font-size: 0.9rem;">Select vehicle specifications above and trigger analysis.</p>
                </div>
                """
            )

        # COLUMN 2: Dark Neon Plotly Line Graph (Matching Revenue Section in Reference)
        with gr.Column(scale=5):
            trend_plot = gr.Plot(value=generate_trend_chart(get_initial_history()), show_label=False)

        # COLUMN 3: Class Distribution Breakdown Chart
        with gr.Column(scale=3, elem_classes=["dashboard-card"]):
            dist_plot = gr.Plot(value=generate_distribution_chart(get_initial_history()), show_label=False)

    # BOTTOM ROW: Recent Assessment History Log (Matching "Booking Recent" Table in Reference)
    with gr.Row():
        with gr.Column(elem_classes=["dashboard-card"]):
            gr.Markdown("### 📋 **Recent Evaluation Log History**")
            
            history_table = gr.Dataframe(
                value=get_initial_history(),
                headers=["Eval_ID", "Timestamp", "Buying", "Maint", "Doors", "Persons", "Boot", "Safety", "Decision", "Status"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                row_count=6
            )

    # Event binding
    btn_run.click(
        fn=process_evaluation,
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety,
            history_state
        ],
        outputs=[
            history_state,
            history_table,
            trend_plot,
            dist_plot,
            result_output,
            kpi_1,
            kpi_2,
            kpi_3,
            kpi_4
        ]
    )

# ==========================================================
# 6. Execution Setup
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Launching Dashboard on port {port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
