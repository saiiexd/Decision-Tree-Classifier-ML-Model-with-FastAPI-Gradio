import subprocess
import time
import sys
import os

def main():
    print("="*40)
    print("   STUDENT PREDICTION SYSTEM   ")
    print("="*40)

    # 1. Initialization
    print("\n[1/3] Initializing ML Model...")
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        print("Model trained and visualization generated.")
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # 2. Start FastAPI Backend
    print("\n[2/3] Starting FastAPI Backend...")
    # Using '127.0.0.1' instead of '0.0.0.0' for better Windows browser compatibility
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for backend to be ready
    time.sleep(4)
    if backend_proc.poll() is not None:
        print("Backend failed to start. Check if port 8000 is occupied.")
        return
    print("Backend running at http://127.0.0.1:8000")

    # 3. Start Gradio Frontend
    print("\n[3/3] Launching Gradio UI...")
    try:
        # Running UI in the main thread so we can catch Ctrl+C easily
        subprocess.run([sys.executable, "ui.py"])
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        backend_proc.terminate()
        print("Backend stopped. Process complete.")

if __name__ == "__main__":
    main()
