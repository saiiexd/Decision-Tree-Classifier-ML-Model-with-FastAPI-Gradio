# 🎓 Student Pass/Fail Predictor 2.0

An upgraded, production-like ML web application using FastAPI, Gradio, and Scikit-Learn.

## ✨ 2.0 Enhancements

- **UI/UX**: HTML coloring (Green/Red), Confidence scores, and a "Reset" button.
- **Explainability**: View the actual Decision Tree logic plot directly in the UI.
- **Robustness**: Pydantic input validation and structured API responses.
- **Insights**: Dynamic feature importance breakdown (see which factor affects the grade most).
- **Health Checks**: New `/health` endpoint for monitoring backend status.

## 🚀 One-Command Launch

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the entire system**:
    ```bash
    python run.py
    ```

## 🛠️ Architecture

- `app.py`: FastAPI backend with validation and `/predict` + `/health` endpoints.
- `ui.py`: Polished Gradio interface with visual feedback.
- `train_model.py`: Trains the logic and generates the decision tree visualization.
- `data_generation.py`: Improved synthetic data with interaction terms for realistic ML training.
- `run.py`: Orchestrator script.

## 📊 Endpoints

- **FastAPI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Gradio UI**: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
