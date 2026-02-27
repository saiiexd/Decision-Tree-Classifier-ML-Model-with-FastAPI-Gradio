import gradio as gr
import requests
import json

def get_prediction(age, salary):
    url = "http://localhost:8000/predict"
    payload = {"age": int(age), "salary": int(salary)}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return f"Result: {result['status']}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Backend connection failed: {str(e)}"

# Define the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌳 Decision Tree Classifier")
    gr.Markdown("Predict if a customer will purchase a product based on Age and Estimated Salary.")
    
    with gr.Row():
        age_input = gr.Number(label="Age", value=30, precision=0)
        salary_input = gr.Number(label="Estimated Salary ($)", value=50000, precision=0)
    
    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Prediction Result")
    
    predict_btn.click(fn=get_prediction, inputs=[age_input, salary_input], outputs=output)
    
    gr.Markdown("### Simple & Subtle Interface")

if __name__ == "__main__":
    demo.launch(server_port=7860)
