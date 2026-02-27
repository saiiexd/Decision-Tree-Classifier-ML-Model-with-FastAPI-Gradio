# Student Pass/Fail Prediction System

This project is a comprehensive machine learning application designed to predict student academic outcomes. It utilizes a Decision Tree Classifier for prediction logic, a FastAPI backend for model serving, and a Gradio frontend for user interaction.

## Overview

The system analyzes three primary factors to determine a student's status:
- Study Hours: Total hours spent studying per day.
- Attendance: Percentage of classes attended.
- Previous Score: The score achieved in the most recent examination.

The model provides a Pass or Fail prediction along with a confidence score and a breakdown of feature importance, showing which factors most influenced the result.

## Project Structure

- `data_generation.py`: Generates a synthetic dataset with realistic student performance patterns.
- `train_model.py`: Trains the Decision Tree model, performs evaluations, and generates a visualization of the decision logic.
- `app.py`: A FastAPI web server that exposes a RESTful endpoint for predictions.
- `ui.py`: A Gradio-based web interface for interactive user input and result visualization.
- `run.py`: An orchestration script to initialize training and launch all services simultaneously.
- `requirements.txt`: Documentation of necessary Python dependencies.
- `utils.py`: Helper functions for model persistence and file handling.

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Dependencies
Install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

### Step 2: Initialize and Run the Application
The entire system can be launched using the provided unified runner:
```bash
python run.py
```

This command will:
1. Generate the training data.
2. Train the Decision Tree model.
3. Save the model and its metadata for serving.
4. Generate a visualization of the decision logic.
5. Launch the FastAPI backend on port 8000.
6. Launch the Gradio frontend on port 7860.

## Usage

Once the application is running, follow these steps to use the system:

1. Open your web browser and navigate to `http://127.0.0.1:7860`.
2. Adjust the sliders to represent a student's study habits and previous scores.
3. Click the "Predict" button to receive a prediction.
4. Review the "How it Works" section to see the visual representation of the model's decision path.
5. Review the feature importance breakdown to understand the weight assigned to each input variable.

## API Documentation

The backend service also provides automated API documentation. While the system is running, you can access:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`
