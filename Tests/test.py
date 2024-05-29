import cv2
import numpy as np
import time
from queue import Queue

target_offset_queue = Queue()

def detector(fps_limit=15, width=640, height=480, debug=True):
    """Capture video frames, detect blue objects, and compute their displacement from the center.
    
    Args:
        fps_limit (int): Frame rate limit for video capture.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        debug (bool): Flag to activate debugging mode which shows output frames.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_limit)

    # Calculate 10% of the frame's area
    min_area = 0.05 * width * height

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define blue color range
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        median = cv2.medianBlur(blue, 15)

        # Find contours
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                # Process the contour if it's large enough
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                center_frame_x = width // 2
                displacement_x = cX - center_frame_x
                target_offset_queue.put(-displacement_x)

                if debug:
                    cv2.putText(frame, f"Displacement: {displacement_x}px", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "center", (cX - 20, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if debug:
            cv2.imshow("Blue", median)
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key
            break

        time.sleep(0.1)

    cap.release()
    if debug:
        cv2.destroyAllWindows()

detector()
