import gradio as gr
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
model = joblib.load("Car_Evaluation.pkl")


# ==========================
# Prediction Function
# ==========================
def predict_car(
    buying,
    maint,
    doors,
    persons,
    lug_boot,
    safety
):

    data = pd.DataFrame({
        "buying": [buying],
        "maint": [maint],
        "doors": [doors],
        "persons": [persons],
        "lug_boot": [lug_boot],
        "safety": [safety]
    })

    prediction = model.predict(data)[0]

    labels = {
        0: "🚗 Unacceptable",
        1: "🙂 Acceptable",
        2: "⭐⭐ Good",
        3: "🏆 Very Good",
        "unacc": "🚗 Unacceptable",
        "acc": "🙂 Acceptable",
        "good": "⭐⭐ Good",
        "vgood": "🏆 Very Good"
    }

    return labels.get(prediction, prediction)


# ==========================
# Gradio UI
# ==========================
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Car Evaluation Prediction System"
) as demo:

    gr.Markdown(
        """
        # 🚘 Car Evaluation Prediction System
        
        Predict the evaluation category of a car using Machine Learning.

        ---
        """
    )

    with gr.Row():

        buying = gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            value="low",
            label="Buying Price"
        )

        maint = gr.Dropdown(
            ["low", "med", "high", "vhigh"],
            value="low",
            label="Maintenance Cost"
        )

    with gr.Row():

        doors = gr.Dropdown(
            ["2", "3", "4", "5more"],
            value="4",
            label="Number of Doors"
        )

        persons = gr.Dropdown(
            ["2", "4", "more"],
            value="4",
            label="Person Capacity"
        )

    with gr.Row():

        lug_boot = gr.Dropdown(
            ["small", "med", "big"],
            value="med",
            label="Luggage Boot Size"
        )

        safety = gr.Dropdown(
            ["low", "med", "high"],
            value="high",
            label="Safety"
        )

    predict_btn = gr.Button(
        "🔍 Predict Car Evaluation",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction",
        lines=1
    )

    predict_btn.click(
        predict_car,
        inputs=[
            buying,
            maint,
            doors,
            persons,
            lug_boot,
            safety
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            ["low", "low", "4", "4", "big", "high"],
            ["vhigh", "vhigh", "2", "2", "small", "low"],
            ["med", "med", "4", "more", "big", "high"],
            ["high", "high", "3", "2", "small", "med"]
        ],
        inputs=[
            buying,
            maint,
            doors,
            persons,
            lug_boot,
            safety
        ]
    )

    gr.Markdown(
        """
        ---
        ### 👨‍💻 Developed by **Sameer**
        **Roll No. - 241020**
        """
    )

demo.launch()
