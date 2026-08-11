```python
import gradio as gr
import joblib
import pandas as pd


# =========================================================
# LOAD HYBRID MODEL
# =========================================================

model = joblib.load("Car_Eva.pkl")


# =========================================================
# MAPPINGS
# These MUST match the mappings used during model training
# =========================================================

buying_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

maintenance_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

doors_map = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5more": 3
}

persons_map = {
    "2": 0,
    "4": 1,
    "more": 2
}

lug_boot_map = {
    "small": 0,
    "med": 1,
    "big": 2
}

safety_map = {
    "low": 0,
    "med": 1,
    "high": 2
}


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_car(
    buying_price,
    maintenance_cost,
    number_of_doors,
    number_of_persons,
    lug_boot,
    safety
):

    # Convert categorical values into the same
    # numerical values used during model training

    data = pd.DataFrame({
        "buying price": [
            buying_map[buying_price]
        ],

        "maintenance cost": [
            maintenance_map[maintenance_cost]
        ],

        "number of doors": [
            doors_map[number_of_doors]
        ],

        "number of persons": [
            persons_map[number_of_persons]
        ],

        "lug_boot": [
            lug_boot_map[lug_boot]
        ],

        "safety": [
            safety_map[safety]
        ]
    })

    # Make prediction
    prediction = model.predict(data)[0]

    # Convert numerical prediction back to class name
    result_map = {
        0: "UNACCEPTABLE",
        1: "ACCEPTABLE",
        2: "GOOD",
        3: "VERY GOOD"
    }

    result = result_map[int(prediction)]

    return f"Car Evaluation: {result}"


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Car Safety and Evaluation Prediction System"
) as demo:

    gr.Markdown(
        """
        # 🚗 Car Safety and Evaluation Prediction System

        ### Hybrid Machine Learning Model
        **XGBoost + Random Forest**

        Enter the car details below to predict its
        overall evaluation.
        """
    )

    gr.Markdown(
        """
        ### 📋 Car Information
        """
    )

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

    predict_button = gr.Button(
        "🔍 Predict Car Evaluation",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result",
        interactive=False
    )

    predict_button.click(
        fn=predict_car,
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ],
        outputs=output
    )

    gr.Markdown(
        """
        ---
        ### 🤖 Model Information

        **Algorithm:** Hybrid Ensemble  
        **Models:** XGBoost + Random Forest  
        **Voting:** Soft Voting  

        **Developed by Sameer**  
        **Roll No.: 241020**
        """
    )


# =========================================================
# LAUNCH APPLICATION
# =========================================================

if __name__ == "__main__":
    demo.launch()
```
