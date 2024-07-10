import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety Helmet Detection")
    parser.add_argument("--webcam-resolution", default=[1080, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()    
    frame_width, frame_height = args.webcam_resolution
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cv2.namedWindow('Safety Helmet Detection', cv2.WINDOW_NORMAL)

    model = YOLO(r'runs/detect/train/100_Epochs/weights/best.pt')

    color = sv.ColorPalette.from_hex(['#32CD32', '#FF5733'])

    box_annotator = sv.RoundBoxAnnotator(thickness=2, color=color)
    label_annotator = sv.LabelAnnotator(color=color)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = model(frame, conf = 0.55, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]
        frame = box_annotator.annotate(scene=frame, detections=detections, )
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = cv2.putText(frame, text=('Workers Without Helmet!' if 'No Helmet' in detections['class_name'] else ''), org=(50, 50), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=3)

        cv2.imshow("Safety Helmet Detection", frame)
        if cv2.waitKey(30) == 27 or cv2.getWindowProperty('Safety Helmet Detection', cv2.WND_PROP_VISIBLE) < 1: # Press Esc to close window
            break
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
