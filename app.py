import gradio as gr
import joblib
import pandas as pd

# Load Model
model = joblib.load("Car_Evaluation.pkl")


def predict_car(buying_price, maintenance_cost, number_of_doors,
                number_of_persons, lug_boot, safety):

    data = pd.DataFrame({
        "buying price": [buying_price],
        "maintenance cost": [maintenance_cost],
        "number of doors": [number_of_doors],
        "number of persons": [number_of_persons],
        "lug_boot": [lug_boot],
        "safety": [safety]
    })

    prediction = model.predict(data)[0]

    return f"🚗 Car Evaluation Result: {prediction}"


css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

*{
    font-family:'Poppins',sans-serif;
}

body{
    background:linear-gradient(135deg,#050816,#0b132b,#101c3d);
    background-attachment:fixed;
}

.gradio-container{
    max-width:1250px !important;
    margin:auto;
    padding:20px;
}

.gr-block,
.gr-box{
    background:rgba(18,28,55,0.75) !important;
    backdrop-filter:blur(15px);
    border:1px solid rgba(0,180,255,.25);
    border-radius:20px !important;
    box-shadow:
    0 0 20px rgba(0,140,255,.15),
    inset 0 0 8px rgba(255,255,255,.04);
}

label{
    color:#ffffff !important;
    font-weight:600 !important;
}

input,
textarea,
select{
    background:#111c36 !important;
    color:white !important;
    border:1px solid #1da1f2 !important;
    border-radius:12px !important;
}

button{
    background:linear-gradient(90deg,#00c6ff,#0072ff) !important;
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-size:18px !important;
    font-weight:700 !important;
    padding:14px !important;
    transition:.35s;
}

button:hover{
    transform:translateY(-3px);
    box-shadow:
    0 0 25px #00bfff,
    0 0 45px rgba(0,191,255,.5);
}

textarea{
    font-size:18px !important;
    font-weight:bold;
}

footer{
    visibility:hidden;
}

h1,h2,h3{
    color:white !important;
}
"""

with gr.Blocks(css=css, title="Car Evaluation System") as demo:

    gr.Markdown("""
    <div style="
    background:linear-gradient(90deg,#001F3F,#0059B3,#0099FF);
    padding:30px;
    border-radius:20px;
    text-align:center;
    box-shadow:0 0 25px rgba(0,170,255,.45);
    ">
    <h1 style="color:white;font-size:42px;">
    🚘 CAR EVALUATION SYSTEM
    </h1>
    <p style="font-size:20px;color:#E8F8FF;">
    Luxury Automobile Dashboard • AI Powered Vehicle Classification
    </p>
    </div>
    """)

    with gr.Row():

        with gr.Column():
            gr.Markdown("## 🚗 Vehicle Specifications")

            buying_price = gr.Dropdown(
                ["low", "med", "high", "vhigh"],
                label="Buying Price",
                value="med"
            )

            maintenance_cost = gr.Dropdown(
                ["low", "med", "high", "vhigh"],
                label="Maintenance Cost",
                value="med"
            )

            number_of_doors = gr.Dropdown(
                ["2", "3", "4", "5more"],
                label="Number of Doors",
                value="4"
            )

            number_of_persons = gr.Dropdown(
                ["2", "4", "more"],
                label="Number of Persons",
                value="4"
            )

            lug_boot = gr.Dropdown(
                ["small", "med", "big"],
                label="Luggage Boot Size",
                value="med"
            )

            safety = gr.Dropdown(
                ["low", "med", "high"],
                label="Safety",
                value="high"
            )

        with gr.Column():
            gr.Markdown("## 📊 AI Evaluation Dashboard")

            output = gr.Textbox(
                label="🏁 Prediction Result",
                lines=4
            )

    predict_btn = gr.Button(
        "🚘 Evaluate Vehicle",
        variant="primary"
    )

    predict_btn.click(
        predict_car,
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

    gr.Examples(
        examples=[
            ["low", "low", "4", "more", "big", "high"],
            ["med", "med", "4", "4", "med", "med"],
            ["high", "high", "2", "2", "small", "low"],
            ["vhigh", "vhigh", "2", "2", "small", "low"]
        ],
        inputs=[
            buying_price,
            maintenance_cost,
            number_of_doors,
            number_of_persons,
            lug_boot,
            safety
        ]
    )

    gr.Markdown("""
    ---
    <div style="text-align:center;color:#A5CFFF">
    <h3>🚗 Developed by Sameer</h3>
    <p><b>Roll No. 241020</b></p>
    <p>Machine Learning Project</p>
    </div>
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)
