from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

def train_model():
    # Load model
    model = YOLO('yolov5s.pt')

    # Train model on GPU (or CPU if GPU is not available)
    results = model.train(data='datasets/drone_dataset/data.yaml', device=device, epochs=100)

if __name__ == "__main__":
    train_model()


