import cv2
import numpy as np
import time
import pigpio
from gpiozero import Button, DistanceSensor
import queue
import threading
import logging

# Define GPIO pins used for motor control and sensor inputs
STEP_PIN = 12 # Hardware PWM only available on GPIO 12, 13
DIR_PIN = 20
SWITCH_PIN = 16
START_PIN = 23
RESET_PIN = 24
TRIG_PIN = 4
ECHO_PIN = 17

# Define constants for navigation and motor operation
SAFE_DIST = 300  # Safe distance threshold from wall in millimeters
REQ_CONSEC = 5  # Required consecutive readings for alignment
X_OFFSET_CONV_FACTOR = 0.15  # Conversion factor for x offset
DATUM_OFFSET = 1900  # Steps to align with datum
CAMERA_ORGIN_OFFSET = -40

# Specification for stopping time at the end of phase one
PHASE_1_STOP_TIME = 7.5

# Initialize a queue to store target offsets detected by the camera
target_offset_queue = queue.Queue()

def detector(target_offset_queue: queue.Queue, stop_event, fps_limit=15, width=640, height=480, debug=False):
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
            red_mask = cv2.medianBlur(red_mask, 5)

            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process largest valid contour
            largest_contour = max(
                (cnt for cnt in contours if cv2.contourArea(cnt) >= min_area), 
                key=cv2.contourArea,
                default=None
            )
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

def distance():
    return ultrasonic.distance * 1000

def move_motor(start_frequency, final_frequency, steps, dir=1, run_time=None, stop=False):
    """Control motor movement with ramping."""
    try:
        # Set the motor direction
        pi.write(DIR_PIN, dir)

        # Determine ramping direction
        if start_frequency < final_frequency:
            # Ramp-Up
            for step in range(steps):
                progress = step / steps
                current_frequency = start_frequency + (final_frequency - start_frequency) * progress
                pi.hardware_PWM(STEP_PIN, int(current_frequency), 500000)
                time.sleep(0.05)
        else:
            # Ramp-Down
            for step in range(steps):
                progress = step / steps
                current_frequency = start_frequency - (start_frequency - final_frequency) * progress
                pi.hardware_PWM(STEP_PIN, int(current_frequency), 500000)
                time.sleep(0.05)

        # Hold phase or continue indefinitely
        if run_time:
            pi.hardware_PWM(STEP_PIN, int(final_frequency), 500000)
            time.sleep(run_time)
            if stop: 
                stop_motor(start_frequency=final_frequency) 
        else:
            # Continue indefinitely at final frequency
            pi.hardware_PWM(STEP_PIN, int(final_frequency), 500000)

    except KeyboardInterrupt:
        logging.info("Motor stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
# Function to stop the motor
def stop_motor(start_frequency=100, steps=20):
    """Stops the motor by ramping down and then turning off the PWM signal."""
    try:
        # Ramp-Down
        for step in range(steps):
            progress = step / steps
            current_frequency = start_frequency - (start_frequency * progress)
            pi.hardware_PWM(STEP_PIN, int(current_frequency), 500000)
            time.sleep(0.05)

        # Stop PWM
        pi.hardware_PWM(STEP_PIN, 0, 0)
        logging.info("Motor stopped successfully.")
    except Exception as e:
        logging.error(f"An error occurred while stopping the motor: {e}")

# Function to align with the target
def align(target_offset_queue: queue.Queue, stop_event, kp=0.1, max_frequency=1000, min_frequency=100, steps=20):
    """Aligns the motor with the target using the displacement detected by the detector."""
    try:
        while not stop_event.is_set():
            if not target_offset_queue.empty():
                displacement = target_offset_queue.get()
                
                # Calculate frequency based on displacement
                frequency = min(max(min_frequency, abs(displacement) * kp), max_frequency)
                direction = 1 if displacement > 0 else 0

                if frequency > min_frequency:
                    move_motor(start_frequency=min_frequency, final_frequency=frequency, steps=steps, dir=direction, run_time=0.1)
                else:
                    stop_motor(start_frequency=frequency, steps=steps)
            else:
                time.sleep(0.01)  # Avoid busy-waiting

    except KeyboardInterrupt:
        logging.info("Alignment stopped by user.")
    except Exception as e:
        logging.error(f"Alignment error: {e}")
    finally:
        stop_motor()
        logging.info("Alignment loop terminated.")

def cycle(target_offset_queue: queue.Queue, stop_event):
    """Control the full operational cycle of the system."""
    try:
        # Start moving forward
        move_motor(start_frequency=10, final_frequency=1000, steps=50, dir=0)
        start_time = time.time()

        # Continuously check distance
        while not stop_event.is_set():
            current_distance = distance()
            logging.info(f"Distance: {current_distance:.1f} mm")
            
            if current_distance <= SAFE_DIST:
                # Slow down as it gets close to the barrier
                move_motor(start_frequency=1000, final_frequency=300, steps=50, dir=0)
                end_time = time.time()
                break

            time.sleep(0.1)

        while not stop_event.is_set():
            if limit_switch.is_pressed:
                stop_motor()
                break
            time.sleep(0.1)

        # Return to the origin
        logging.info("Moving back")
        move_motor(start_frequency=10, final_frequency=1000, steps=50, dir=1, run_time=(end_time - start_time))
        move_motor(start_frequency=1000, final_frequency=300, steps=50, dir=1)
        
        logging.info('Finding target')
        while not stop_event.is_set():
            if not target_offset_queue.empty():
                break
        logging.info("Found target")

        # Align with the origin / target
        logging.info("Aligning")
        align(target_offset_queue, stop_event)
        
        pi.write(STEP_PIN, 0)
        logging.info("Camera aligned with target")

        # Align with datum
        move_motor(start_frequency=10, final_frequency=200, steps=100, dir=0, run_time=DATUM_OFFSET / 200, stop=True)
        pi.write(STEP_PIN, 0)
        logging.info("Aligned")

        time.sleep(PHASE_1_STOP_TIME)

        # Move forward to find the next target
        move_motor(start_frequency=100, final_frequency=1000, steps=50, dir=0)
        time.sleep(10)

        while not stop_event.is_set():
            if not target_offset_queue.empty():
                move_motor(start_frequency=1000, final_frequency=300, steps=100, dir=0)
                break
        
        logging.info("Aligning")
        align(target_offset_queue, stop_event)
        
        pi.write(STEP_PIN, 0)
        logging.info("Camera aligned with target")

        # Align with datum
        move_motor(start_frequency=10, final_frequency=200, steps=100, dir=0, run_time=DATUM_OFFSET / 200, stop=True)
        pi.write(STEP_PIN, 0)
        logging.info("Aligned")

        logging.info("Cycle complete")
        time.sleep(20)

    except Exception as e:
        logging.error(f"Cycle error: {str(e)}")
        stop_motor()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize pigpio library instance and configure GPIO modes
    pi = pigpio.pi()

    if not pi.connected:
        logging.error("Error connecting to pigpio daemon. Is the daemon running?")
        exit(1)
    
    pi.set_mode(STEP_PIN, pigpio.OUTPUT)
    pi.set_mode(DIR_PIN, pigpio.OUTPUT)

    # Initialize sensors and input devices
    ultrasonic = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN)
    limit_switch = Button(SWITCH_PIN)
    start = Button(START_PIN)
    reset = Button(RESET_PIN)

    # Initialize thread-safe components
    offset_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    try:
        detector_thread = threading.Thread(
            target=detector,
            args=(offset_queue, stop_event)
        )
        detector_thread.start()
        
        cycle_thread = threading.Thread(
            target=cycle,
            args=(offset_queue, stop_event)
        )
        cycle_thread.start()

        # Wait for threads to complete
        detector_thread.join()
        cycle_thread.join()

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected, stopping all threads.")
    finally:
        stop_event.set()
        pi.stop()  # Clean up pigpio resources