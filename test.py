from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\DJATO\YOLO\ultralytics\runs\detect\drone_detector29\weights\last.pt')  # path to your best model

# Run inference on various sources
results = model.predict(source=r'C:\Users\DJATO\Desktop\dose-media-DiTiYQx0mh4-unsplash.jpg', save=True)  # image
#results = model.predict(source='path/to/video.mp4')  # video
#results = model.predict(source='path/to/folder')  # directory with images
#results = model.predict(source=0)  # webcam

# Customize inference parameters

