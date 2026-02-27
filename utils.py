import pickle
import os

def check_assets():
    """Checks if model and metadata files exist."""
    return os.path.exists("model.pkl") and os.path.exists("metadata.pkl")

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
