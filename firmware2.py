import cv2
import numpy as np
import time
from queue import Queue
import queue
import logging

def detector(target_offset_queue: Queue, stop_event, fps_limit=15, width=640, height=480, debug=False):
    """
    Thread-safe function to capture video frames and detect blue objects.
    
    Args:
        target_offset_queue (Queue): Thread-safe queue to store displacement values
        stop_event: Threading event to signal stopping
        fps_limit (int): Frame rate limit for video capture
        width (int): Width of the video frame
        height (int): Height of the video frame
        debug (bool): Flag to activate debugging mode
    """
    try:
        # Initialise camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        cap.set(cv2.CAP_PROP_FPS, fps_limit)

        # Precalculate constants
        min_area = 0.05 * width * height
        center_frame_x = width // 2
        frame_time = 1 / fps_limit
        last_frame_time = time.time()

        while not stop_event.is_set():
            # Frame rate control
            current_time = time.time()
            if (current_time - last_frame_time) < frame_time:
                time.sleep(0.001)
                continue
            
            last_frame_time = current_time

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame")
                continue

            # Resize frame to reduce processing load
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # Convert to LAB and threshold directly on the 'a' channel
            b, g, r = cv2.split(frame)
            red_mask = cv2.threshold(
                cv2.subtract(r, cv2.max(b, g)),  # R - max(B,G)
                30,  # Threshold value
                255,
                cv2.THRESH_BINARY
            )[1]

            # Apply blur
            red_mask = cv2.medianBlur(red_mask, 10)

            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            largest_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area and area > largest_area:
                    largest_area = area
                    largest_contour = cnt

            if largest_contour is not None:
                # Use bounding box for displacement
                x, y, w, h = cv2.boundingRect(largest_contour)
                cX = x + w // 2
                displacement_x = cX - center_frame_x

                # Non-blocking queue operation
                try:
                    target_offset_queue.put_nowait(-displacement_x)
                except Exception as e:
                    logging.warning(f"Queue operation failed: {e}")

                if debug:
                    # Draw debug information
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Displacement: {displacement_x}px",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if debug:
                cv2.imshow("Frame", frame)
                cv2.imshow("Mask", red_mask)

                if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
                    break

    except Exception as e:
        logging.error(f"Detector error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        if debug:
            cv2.destroyAllWindows()

# Example usage in a multi-threaded context:
if __name__ == "__main__":
    import threading
    
    # Initialize thread-safe components
    offset_queue = Queue(maxsize=20)
    stop_event = threading.Event()
    
    # Start detector in a thread
    detector_thread = threading.Thread(
        target=detector,
        args=(offset_queue, stop_event),
        kwargs={'debug': True}
    )
    detector_thread.start()
    
    try:
        while True:
            # Process offsets in main thread
            try:
                offset = offset_queue.get(timeout=0.1)
                print(f"Offset: {offset}")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        stop_event.set()
        detector_thread.join()