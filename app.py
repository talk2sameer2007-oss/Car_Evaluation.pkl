# app.py

import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
# --- CODE BLOCK: LOAD XGBOOST MODEL ---
try:
    deployed_xgb = joblib.load("Car_Evaluation.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_xgb = None
# --------------------------------------

# ==========================================================
# Prediction Function with Bulletproof Error Handling
# ==========================================================
# --- CODE BLOCK: 6-FEATURE PREDICTION LOGIC ---
def predict_car_safety(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):
    # Capture inputs from Gradio
    values = [
        buying_price, maintenance_cost, number_of_doors, 
        number_of_persons, lug_boot, safety
    ]

    # 1. Empty/None input check (Bulletproof catch if user skips a dropdown)
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please select an option for all input fields."

    # 2. Type casting to integers
    try:
        buying_price = int(buying_price)
        maintenance_cost = int(maintenance_cost)
        number_of_doors = int(number_of_doors)
        number_of_persons = int(number_of_persons)
        lug_boot = int(lug_boot)
        safety = int(safety)
    except (ValueError, TypeError):
        return "❌ Internal Error: Invalid data format received."

    # 3. Model execution
    if deployed_xgb is None:
        return "❌ Model failed to load. Please check your .pkl file."

    try:
        # Array strictly ordered to match the X dataframe provided
        input_data = [[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ]]

        prediction = deployed_xgb.predict(input_data)

        # Assuming standard label encoding for the 'decision' target
        # (Modify these return strings if your dataset used different target labels like unacc, acc, good, vgood)
        result_map = {
            0: "Unacceptable (unacc)",
            1: "Acceptable (acc)",
            2: "Good (good)",
            3: "Very Good (vgood)"
        }
        
        # Fallback to the raw prediction if it doesn't match 0-3
        final_decision = result_map.get(prediction[0], f"Class {prediction[0]}")

        return f"🚙 Evaluation Result\n\nCar Safety Decision: {final_decision}"

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"
# ----------------------------------------------

# ==========================================================
# Description & Footer
# ==========================================================
# --- CODE BLOCK: BRANDING & UI TEXT ---
DESCRIPTION = """
# 🚙 Car Safety & Evaluation System

This application evaluates a vehicle's overall acceptability based on its physical attributes, pricing, and safety metrics using a trained **XGBoost Machine Learning Model**.

Select the vehicle's specifications below to run the assessment.
"""

developer_info = """
### About the Developer
**Created by:** Sameer

---
### 🛠️ Tools & Technologies Used
* **Machine Learning:** XGBoost Classifier
* **Web Framework:** Gradio
* **Language:** Python
* **Deployment:** Render
"""
# --------------------------------------

# ==========================================================
# Interface Setup
# ==========================================================
# --- CODE BLOCK: SAFE DROPDOWN INPUTS ---
interface = gr.Interface(
    fn=predict_car_safety,
    inputs=[
        gr.Dropdown(choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)], label="Buying Price"),
        gr.Dropdown(choices=[("Low", 0), ("Medium", 1), ("High", 2), ("Very High", 3)], label="Maintenance Cost"),
        gr.Dropdown(choices=[("2", 2), ("3", 3), ("4", 4), ("5 or More", 5)], label="Number of Doors"),
        gr.Dropdown(choices=[("2", 2), ("4", 4), ("More", 5)], label="Number of Persons"),
        gr.Dropdown(choices=[("Small", 0), ("Medium", 1), ("Big", 2)], label="Luggage Boot Size"),
        gr.Dropdown(choices=[("Low", 0), ("Medium", 1), ("High", 2)], label="Safety Rating"),
    ],
    outputs=gr.Textbox(label="Assessment Result", lines=4),
    title="🚙 Car Safety Evaluation System",
    description=DESCRIPTION,
    article=developer_info
)
# ----------------------------------------

# ==========================================================
# Launch Configuration
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Gradio server on 0.0.0.0:{port}...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
