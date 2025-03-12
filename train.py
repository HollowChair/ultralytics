from ultralytics import YOLO
from multiprocessing import freeze_support

def train():
    # Load a pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # Start with a smaller model for faster training

    # Train the model on your custom dataset
    results = model.train(
        data='data.yaml',
        epochs=100,           # Adjust based on your needs
        imgsz=640,            # Image size
        batch=16,             # Adjust based on your GPU
        patience=20,          # Early stopping patience
        name='drone_detector', # Name your experiment
    )

    # Evaluate the model
    model.val()

if __name__ == '__main__':
    freeze_support()
    train()