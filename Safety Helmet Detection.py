from ultralytics import YOLO

# To make predictions on media files :
model = YOLO(r'runs/detect/train/100_Epochs/weights/best.pt')
model.predict('sample/sample_video_1.mp4', save=True, conf=0.55, agnostic_nms=True)