import gradio as gr
import requests
import os
import time

def get_prediction(study_hours, attendance, previous_score):
    """Hits the FastAPI backend and processes the result."""
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            res = response.json()
            pred = res["prediction"]
            conf = res["confidence"]
            importance = res["feature_importance"]
            
            # Styling based on prediction
            color = "#28a745" if pred == "Pass" else "#dc3545"
            
            # Formatted Result HTML
            result_html = f"""
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="margin: 0;">Predicted Status: {pred}</h2>
                <h3 style="margin: 5px 0 0 0;">Confidence: {conf*100:.0f}%</h3>
            </div>
            """
            
            # Importance breakdown
            imp_text = "### Feature Importance Breakdown\n"
            for k, v in importance.items():
                imp_text += f"- **{k.replace('_', ' ').title()}**: {v*100:.1f}%\n"
                
            return result_html, imp_text
        else:
            return f"❌ Error: {response.text}", ""
    except Exception as e:
        return f"❌ Backend Connection Error: {str(e)}", ""

def reset_inputs():
    """Resets all sliders and outputs."""
    return 6.0, 75.0, 50.0, "", ""

# --- Gradio UI Layout ---
with gr.Blocks(title="Student Performance Predictor 2.0") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 10px;">
        <h1 style="color: #2D3E50;">🎓 Student Pass/Fail Predictor 2.0</h1>
        <p style="font-size: 1.1em; color: #5F6368;">Predict academic outcomes using machine learning logic.</p>
    </div>
    """)
    
    with gr.Row():
        # LEFT COLUMN: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Input Parameters")
            study_slider = gr.Slider(0, 12, value=6, step=0.1, label="Study Hours / Day")
            attendance_slider = gr.Slider(0, 100, value=75, step=1, label="Attendance %")
            score_slider = gr.Slider(0, 100, value=50, step=1, label="Previous Exam Score")
            
            with gr.Row():
                predict_btn = gr.Button("🚀 Predict Now", variant="primary")
                reset_btn = gr.Button("🔄 Reset Inputs")

        # RIGHT COLUMN: Results & Insights
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")
            result_display = gr.HTML(label="Result Area")
            importance_display = gr.Markdown(label="Feature Importance")

    # SECOND ROW: Visualization & Explanation
    with gr.Row():
        with gr.Accordion("🧠 How it Works: The Decision Tree", open=False):
            gr.Markdown("""
            The model uses a **Decision Tree Classifier** to find patterns in student behavior. 
            It asks a series of 'Yes/No' questions (e.g., *Is attendance > 80%?*) to reach a final prediction.
            Below is the actual visualization of the logic currently powering this system:
            """)
            if os.path.exists("tree_plot.png"):
                gr.Image("tree_plot.png", label="Decision Tree Logic Path")
            else:
                gr.Markdown("*(Tree visualization will appear after the first model training)*")

    # --- Interactions ---
    predict_btn.click(
        fn=get_prediction,
        inputs=[study_slider, attendance_slider, score_slider],
        outputs=[result_display, importance_display]
    )
    
    reset_btn.click(
        fn=reset_inputs,
        inputs=[],
        outputs=[study_slider, attendance_slider, score_slider, result_display, importance_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
