import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker 

model = YOLO('yolov8s.pt')


def calculate_speed(displacement, frame_rate, pixel_per_meter=None):
    """Calculates speed in km/h based on displacement and frame rate.

    Args:
        displacement (int): The displacement in pixels between consecutive frames.
        frame_rate (float): The video frame rate.
        pixel_per_meter (float, optional): The conversion factor from pixels to meters
            for real-world speed calculation. Defaults to None (no conversion).

    Returns:
        float: The speed in km/h (or pixels/frame if pixel_per_meter is None).
    """
    conversion_factor = 3.6 
    if pixel_per_meter is None:
        speed = displacement / frame_rate
    else:
        
        meter_displacement = displacement / pixel_per_meter
        speed = meter_displacement * frame_rate * conversion_factor
    return speed


def main():
    
    n = 1  

    
    cap = cv2.VideoCapture('veh2.mp4')
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    
    class_list = []
    with open("coco.txt", "r") as f:
        class_list = f.read().splitlines()

    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    tracker = Tracker()

    
    prev_cx = {}


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model.predict(frame)

        
        if not hasattr(results, 'boxes'):
            print("Error: 'boxes' attribute not found in results. Skipping frame.")
            continue

        
        boxes_data = results.boxes.data.to('cpu').numpy().astype("int")
        car_boxes = []
        for box in boxes_data:
            x1, y1, x2, y2, confidence = box
            car_boxes.append([x1, y1, x2, y2])

        
        bbox_id = tracker.update(car_boxes)

        
        for bbox, id in zip(bbox_id, tracker.tracks.keys()):
            if len(bbox) < 4:
                
                print(f"Error: Insufficient values in bbox for ID {id}. Skipping.")
                continue
            
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            
            displacement = abs(cx - prev_cx.get(id, cx))
            prev_cx[id] = cx

            
            speed = calculate_speed(displacement, frame_rate)

            
            cv2.putText(frame, f"ID: {id}, Speed: {speed:.2f} km/h", (cx, cy),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        
        cv2.imshow("Speed Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 113:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
