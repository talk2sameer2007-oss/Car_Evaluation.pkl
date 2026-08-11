import os
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import gradio as gr

# ==========================================================
# 1. Model Configuration & Feature Mappings
# ==========================================================
MODEL_PATH = "Car_Eva.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Machine Learning Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model load note: {e}. Running in simulation fallback mode.")
    model = None

# Input Category Mappings
buying_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
maintenance_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
doors_map = {"2": 0, "3": 1, "4": 2, "5more": 3}
persons_map = {"2": 0, "4": 1, "more": 2}
lug_boot_map = {"small": 0, "med": 1, "big": 2}
safety_map = {"low": 0, "med": 1, "high": 2}

RESULT_MAP = {
    0: ("UNACCEPTABLE", "FAIL", "#ef4444"),
    1: ("ACCEPTABLE", "PASS", "#f59e0b"),
    2: ("GOOD", "PASS", "#3b82f6"),
    3: ("VERY GOOD", "PASS", "#10b981")
}

def get_initial_history():
    return pd.DataFrame([
        {"Eval_ID": "EV-1001", "Time": "10:15:20", "Buying": "med", "Maint": "low", "Doors": "4", "Persons": "4", "Boot": "med", "Safety": "high", "Decision": "VERY GOOD", "Status": "PASS"},
        {"Eval_ID": "EV-1002", "Time": "10:32:10", "Buying": "high", "Maint": "high", "Doors": "2", "Persons": "2", "Boot": "small", "Safety": "low", "Decision": "UNACCEPTABLE", "Status": "FAIL"},
        {"Eval_ID": "EV-1003", "Time": "11:05:45", "Buying": "low", "Maint": "med", "Doors": "4", "Persons": "4", "Boot": "big", "Safety": "med", "Decision": "ACCEPTABLE", "Status": "PASS"},
        {"Eval_ID": "EV-1004", "Time": "11:42:12", "Buying": "med", "Maint": "med", "Doors": "5more", "Persons": "more", "Boot": "big", "Safety": "high", "Decision": "GOOD", "Status": "PASS"},
    ])

# ==========================================================
# 2. Analytics Visualization
# ==========================================================
def generate_prediction_chart(safety_val, boot_val):
    categories = ['Price Index', 'Maintenance', 'Door Config', 'Seating', 'Boot Vol.', 'Safety Rating']
    values = [50, 50, 75, 75, (boot_val + 1) * 33, (safety_val + 1) * 33]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.25)',
        line=dict(color='#3b82f6', width=2),
        name='Vehicle Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'),
            angularaxis=dict(tickfont=dict(size=10, color='#94a3b8'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=25, r=25, t=30, b=25),
        height=260,
        showlegend=False,
        title=dict(text="<b>BMW Spec Assessment Radar</b>", font=dict(size=12, color="#f8fafc"))
    )
    return fig

def generate_trend_chart(df_history):
    score_mapping = {"UNACCEPTABLE": 1, "ACCEPTABLE": 2, "GOOD": 3, "VERY GOOD": 4}
    y_values = [score_mapping.get(d, 1) for d in df_history["Decision"]]
    x_values = df_history["Eval_ID"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        line=dict(color='#8b5cf6', width=3, shape='spline'),
        marker=dict(size=8, color='#10b981', line=dict(color='#38bdf8', width=2)),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.15)'
    ))

    fig.update_layout(
        title=dict(text="<b>Telemetry Assessment History</b>", font=dict(color="#f8fafc", size=12)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=35, b=20),
        height=260,
        xaxis=dict(showgrid=False, color='#64748b', tickfont=dict(size=10)),
        yaxis=dict(
            showgrid=True, gridcolor='#1e293b', color='#64748b',
            tickvals=[1, 2, 3, 4], ticktext=['Unacc', 'Acc', 'Good', 'VGood']
        ),
        showlegend=False
    )
    return fig

def create_kpi_card(title, value, subtitle, color="#3b82f6"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub" style="color: {color};">{subtitle}</div>
    </div>
    """

# ==========================================================
# 3. Prediction Handler Logic
# ==========================================================
def process_evaluation(buying_price, maintenance_cost, number_of_doors, number_of_persons, lug_boot, safety, history_df):
    
    # 1. Map input features to DataFrame structure expected by model
    input_data = pd.DataFrame({
        "buying price": [buying_map[buying_price]],
        "maintenance cost": [maintenance_map[maintenance_cost]],
        "number of doors": [doors_map[number_of_doors]],
        "number of persons": [persons_map[number_of_persons]],
        "lug_boot": [lug_boot_map[lug_boot]],
        "safety": [safety_map[safety]]
    })

    # 2. Perform Prediction
    if model is not None:
        try:
            pred_class = int(model.predict(input_data)[0])
        except Exception:
            pred_class = 0 if safety_map[safety] == 0 else 1
    else:
        # Fallback simulation logic if model file is missing
        if safety_map[safety] == 0 or persons_map[number_of_persons] == 0:
            pred_class = 0
        elif safety_map[safety] == 2 and buying_map[buying_price] <= 1 and maintenance_map[maintenance_cost] <= 1:
            pred_class = 3
        elif safety_map[safety] >= 1 and buying_map[buying_price] <= 2:
            pred_class = 2
        else:
            pred_class = 1

    decision_text, status_badge, badge_color = RESULT_MAP.get(pred_class, ("UNACCEPTABLE", "FAIL", "#ef4444"))
    
    eval_id = f"EV-{1001 + len(history_df)}"
    time_str = datetime.now().strftime("%H:%M:%S")

    # 3. Append to Evaluation History
    new_entry = {
        "Eval_ID": eval_id,
        "Time": time_str,
        "Buying": buying_price,
        "Maint": maintenance_cost,
        "Doors": number_of_doors,
        "Persons": number_of_persons,
        "Boot": lug_boot,
        "Safety": safety,
        "Decision": decision_text,
        "Status": status_badge
    }

    updated_df = pd.concat([pd.DataFrame([new_entry]), history_df], ignore_index=True)

    # 4. Re-calculate Dashboard Metrics
    total_evals = len(updated_df)
    pass_cnt = len(updated_df[updated_df["Status"] == "PASS"])
    pass_rate = f"{(pass_cnt / total_evals) * 100:.1f}%"
    high_safety = len(updated_df[updated_df["Safety"] == "high"])

    kpi1 = create_kpi_card("Total Evaluated", f"{total_evals} Vehicles", "↗ Real-time Session", "#38bdf8")
    kpi2 = create_kpi_card("Safety Pass Rate", pass_rate, f"↗ {pass_cnt} Qualified", "#10b981")
    kpi3 = create_kpi_card("High Safety Tier", f"{high_safety} Units", "↗ High Rating", "#8b5cf6")
    kpi4 = create_kpi_card("Latest Evaluation", decision_text, f"Status: {status_badge}", badge_color)

    result_html = f"""
    <div class="result-card" style="border-left: 5px solid {badge_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">PREDICTION RESULT</span>
            <span class="badge" style="background: {badge_color}; color: #ffffff;">{status_badge}</span>
        </div>
        <h2 style="margin: 6px 0; color: {badge_color}; font-size: 1.4rem; font-weight: 800;">Car Evaluation: {decision_text}</h2>
        <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Evaluated under ID <b>{eval_id}</b> at {time_str}.</p>
    </div>
    """

    spec_chart = generate_prediction_chart(safety_map[safety], lug_boot_map[lug_boot])
    trend_chart = generate_trend_chart(updated_df)

    return updated_df, updated_df, spec_chart, trend_chart, result_html, kpi1, kpi2, kpi3, kpi4

# ==========================================================
# 4. Custom Styling & HTML Templates (Luxury Dark Theme)
# ==========================================================
SHOWROOM_CSS = """
:root {
    --bg-color: #0b0f17;
    --card-bg: rgba(18, 24, 38, 0.75);
    --border-color: rgba(255, 255, 255, 0.08);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

body, .gradio-container {
    background-color: var(--bg-color) !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(124, 58, 237, 0.12) 0px, transparent 50%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-main) !important;
}

.top-header {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 20px 30px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.main-title {
    font-size: 1.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.dev-badge {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 8px 18px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1;
}

.kpi-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.kpi-title {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    margin: 4px 0;
}

.kpi-sub {
    font-size: 0.75rem;
    font-weight: 600;
}

.dashboard-panel {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.btn-eval {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    padding: 14px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.3s ease !important;
}

.btn-eval:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6) !important;
}

.result-card {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 14px;
    padding: 18px;
    margin-top: 15px;
    border: 1px solid var(--border-color);
}

.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 800;
}

/* Dropdown & Form Customization */
.gr-box, .gr-input, select {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border-color: var(--border-color) !important;
    color: #f8fafc !important;
}

/* Dataframe Styling */
.gr-dataframe {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-color) !important;
    color: #f8fafc !important;
}
"""

BMW_3D_IFRAME_HTML = """
<iframe srcdoc='
<!DOCTYPE html>
<html>
<head>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
  <style>
    body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: radial-gradient(circle at center, #1e293b 0%, #090d16 100%); font-family: sans-serif; }
    model-viewer { width: 100%; height: 100%; }
    .overlay-badge { position: absolute; top: 14px; left: 14px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.15); color: #38bdf8; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 700; backdrop-filter: blur(8px); z-index: 100; }
    .overlay-hint { position: absolute; bottom: 14px; right: 14px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.15); color: #cbd5e1; padding: 6px 14px; border-radius: 8px; font-size: 11px; font-weight: 700; backdrop-filter: blur(8px); z-index: 100; }
  </style>
</head>
<body>
  <div class="overlay-badge">🏎️ BMW M4 Coupe — 360° Interactive Studio</div>
  <div class="overlay-hint">🖱️ Click & Drag to Rotate 3D Model</div>
  <model-viewer 
    src="https://cdn.jsdelivr.net/gh/fazil47/assets@master/3d/vehicles/bmw_m4_2021.glb" 
    alt="3D BMW M4 Coupe" 
    auto-rotate 
    camera-controls 
    shadow-intensity="1.8" 
    exposure="1.1"
    interaction-prompt="none">
  </model-viewer>
</body>
</html>
' style="width: 100%; height: 480px; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px;"></iframe>
"""

# ==========================================================
# 5. Gradio Dashboard Construction
# ==========================================================
with gr.Blocks(title="Car Safety and Evaluation Prediction System", css=SHOWROOM_CSS) as demo:
    
    history_state = gr.State(get_initial_history())

    # Top Header
    gr.HTML(
        """
        <div class="top-header">
            <div>
                <div class="main-title">🚗 CAR SAFETY AND EVALUATION PREDICTION SYSTEM</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Hybrid Model: <b>XGBoost + Random Forest</b> (Soft Voting)
                </div>
            </div>
            <div class="dev-badge">
                👤 Developer: <b style="color:#38bdf8;">Sameer</b> &nbsp;|&nbsp; Roll No.: <b style="color:#c084fc;">241020</b>
            </div>
        </div>
        """
    )

    # Top KPI Metrics
    with gr.Row():
        kpi_1 = gr.HTML(create_kpi_card("Total Evaluated", "4 Vehicles", "↗ Real-time Session", "#38bdf8"))
        kpi_2 = gr.HTML(create_kpi_card("Safety Pass Rate", "75.0%", "↗ 3 Qualified", "#10b981"))
        kpi_3 = gr.HTML(create_kpi_card("High Safety Tier", "2 Units", "↗ High Rating", "#8b5cf6"))
        kpi_4 = gr.HTML(create_kpi_card("Latest Evaluation", "GOOD", "Status: PASS", "#38bdf8"))

    # Full-Width 3D BMW Interactive Studio
    with gr.Row():
        with gr.Column(elem_classes=["dashboard-panel"]):
            gr.Markdown("### 🏎️ **BMW M4 Interactive 3D Model Studio**")
            gr.HTML(BMW_3D_IFRAME_HTML)

    # Input Control Console & Analytics
    with gr.Row():
        with gr.Column(scale=4, elem_classes=["dashboard-panel"]):
            gr.Markdown("### 🎛️ **Vehicle Specifications Console**")
            
            with gr.Row():
                buying_price = gr.Dropdown(
                    choices=["low", "med", "high", "vhigh"],
                    label="Buying Price",
                    value="med"
                )
                maintenance_cost = gr.Dropdown(
                    choices=["low", "med", "high", "vhigh"],
                    label="Maintenance Cost",
                    value="med"
                )

            with gr.Row():
                number_of_doors = gr.Dropdown(
                    choices=["2", "3", "4", "5more"],
                    label="Number of Doors",
                    value="4"
                )
                number_of_persons = gr.Dropdown(
                    choices=["2", "4", "more"],
                    label="Number of Persons",
                    value="4"
                )

            with gr.Row():
                lug_boot = gr.Dropdown(
                    choices=["small", "med", "big"],
                    label="Luggage Boot",
                    value="big"
                )
                safety = gr.Dropdown(
                    choices=["low", "med", "high"],
                    label="Safety",
                    value="high"
                )

            predict_button = gr.Button("🔍 Predict Car Evaluation ⚡", elem_classes=["btn-eval"])

            result_display = gr.HTML(
                value="""
                <div class="result-card">
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700;">SYSTEM READY</span>
                    <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.88rem;">Configure vehicle specifications above and click predict.</p>
                </div>
                """
            )

        with gr.Column(scale=4, elem_classes=["dashboard-panel"]):
            spec_plot = gr.Plot(value=generate_prediction_chart(2, 2), show_label=False)

        with gr.Column(scale=4, elem_classes=["dashboard-panel"]):
            trend_plot = gr.Plot(value=generate_trend_chart(get_initial_history()), show_label=False)

    # Telemetry Log Table
    with gr.Row():
        with gr.Column(elem_classes=["dashboard-panel"]):
            gr.Markdown("### 📋 **Real-Time Assessment Log History**")
            
            history_table = gr.Dataframe(
                value=get_initial_history(),
                headers=["Eval_ID", "Time", "Buying", "Maint", "Doors", "Persons", "Boot", "Safety", "Decision", "Status"],
                interactive=False,
                row_count=5
            )

    predict_button.click(
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
            spec_plot,
            trend_plot,
            result_display,
            kpi_1,
            kpi_2,
            kpi_3,
            kpi_4
        ]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Launching Dashboard on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port)
