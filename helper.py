from ultralytics import YOLO
import streamlit as st

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    model = YOLO(model_path)
    return model