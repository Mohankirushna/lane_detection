import cv2
import numpy as np
import math

def compute_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

def region_of_interest(image):
    height, width = image.shape[:2]
    polygon = np.array([[
        (0, height),
        (width // 2 - 100, height // 2 + 50),
        (width // 2 + 100, height // 2 + 50),
        (width, height)
    ]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, (255, 255, 255))
    return cv2.bitwise_and(image, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=200)
    return lines

def merge_lines(left_line, right_line, threshold=50):
    if left_line is None or right_line is None:
        return left_line, right_line

    x1_left, y1_left, x2_left, y2_left = left_line[0]
    x1_right, y1_right, x2_right, y2_right = right_line[0]

    distance = abs(x1_left - x1_right)
    
    if distance < threshold:
        x1_merged = (x1_left + x1_right) // 2
        y1_merged = min(y1_left, y1_right)
        x2_merged = (x2_left + x2_right) // 2
        y2_merged = max(y2_left, y2_right)
        merged_line = np.array([x1_merged, y1_merged, x2_merged, y2_merged]).reshape(1, -1)
        return merged_line, None

    return left_line, right_line

def overlay_lanes(frame, lines):
    if lines is not None:
        leftan = None
        left_lane_lines = []
        right_lane_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if slope < 0:
                left_lane_lines.append(line)
            else:
                right_lane_lines.append(line)

        left_lane_lines, right_lane_lines = merge_lines(left_lane_lines[0] if left_lane_lines else None, 
                                                         right_lane_lines[0] if right_lane_lines else None)

        if left_lane_lines is not None:
            x1, y1, x2, y2 = left_lane_lines[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            angle = compute_angle(left_lane_lines)
            leftan = compute_angle(left_lane_lines)
            cv2.putText(frame, f'{angle:.2f}deg', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if right_lane_lines is not None:
            x1, y1, x2, y2 = right_lane_lines[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if leftan:
                angle = int(leftan+compute_angle(right_lane_lines))
            else:
                angle = 0
            cv2.putText(frame, f'{angle}deg', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        lines = detect_lanes(frame)
        frame_with_lanes = overlay_lanes(frame, lines)
        
        cv2.imshow('Lane Detection', frame_with_lanes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = '/Users/mohankirushna.r/Downloads/4K Scenic Byway 12 _ All American Road in Utah, USA - 5 Hour of Road Drive with Relaxing Music.mp4'
    process_video(input_video_path)