import cv2
import numpy as np
import time
import pigpio
from gpiozero import Button, DistanceSensor
import queue
import threading

# Define GPIO pins used for motor control and sensor inputs
STEP_PIN = 21
DIR_PIN = 20

# Define constants for navigation and motor operation
SAFE_DIST = 150  # Safe distance threshold in millimeters
TARGET_CLEARANCE_TIME = 150  # Time to clear target in milliseconds
X_OFFSET_CONV_FACTOR = 0.15  # Conversion factor for x offset
DATUM_OFFSET = 2100  # Steps to align with datum
REQ_CONSEC = 5 # Required consecutive readings for alignment
CAMERA_ORGIN_OFFSET = -40

# Specification for stopping time at the end of phase one
PHASE_1_STOP_TIME = 7.5

# Initialize a queue to store target offsets detected by the camera
target_offset_queue = queue.Queue()

def detector(fps_limit=15, width=640, height=480, debug=False):
    """Capture video frames, detect blue objects, and compute their displacement from the center.
    
    Args:
        fps_limit (int): Frame rate limit for video capture.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        debug (bool): Flag to activate debugging mode which shows output frames.
    """
    cap = cv2.VideoCapture(-1)
    
    # Set the properties for resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set the frame rate limit
    cap.set(cv2.CAP_PROP_FPS, fps_limit)

    while True:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue color
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        median = cv2.medianBlur(blue, 15)

        # Find contours
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the moments to calculate the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Calculate x displacement from the center of the frame
            center_frame_x = width // 2
            displacement_x = cX - center_frame_x
            target_offset_queue.put(-displacement_x)

            if debug:
                # Display the displacement on the frame
                cv2.putText(frame, f"Displacement: {displacement_x}px", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw the largest contour and its center
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(frame, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Clear the queue if no contours are found
            while not target_offset_queue.empty():
                target_offset_queue.get()

        if debug:
            cv2.imshow("Blue", median)
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key
            break

    cap.release()
    if debug:
        cv2.destroyAllWindows()

def move_motor(start_frequency, final_frequency, steps, dir=1, run_time=None):
    """Generate ramp waveforms to control motor speed from a starting to a final frequency.

    Args:
        start_frequency (int): Starting frequency of the ramp.
        final_frequency (int): Ending frequency of the ramp.
        steps (int): Number of steps in the ramp.
        dir (int): Direction to move the motor (0 or 1).
        run_time (float, optional): Time in seconds to run the motor at final frequency.
    """
    if not pi.connected:
        print("Error connecting to pigpio daemon. Is the daemon running?")
        return

    pi.wave_clear()  # clear existing waves
    pi.write(DIR_PIN, dir)

    # Calculate frequency increments
    frequency_step = (final_frequency - start_frequency) / steps
    current_frequency = start_frequency

    wid = []

    for _ in range(steps):
        micros = int(500000 / current_frequency)  # microseconds for half a step
        wf = [
            pigpio.pulse(1 << STEP_PIN, 0, micros),
            pigpio.pulse(0, 1 << STEP_PIN, micros)
        ]
        pi.wave_add_generic(wf)
        wave_id = pi.wave_create()
        wid.append(wave_id)  # Append the new wave ID to the list
        current_frequency += frequency_step  # increment or decrement frequency
    
    # Generate a chain of waves
    chain = []
    for wave_id in wid:
        chain += [255, 0, wave_id, 255, 1, 1, 0]  # Transmit each wave once

    pi.wave_chain(chain)  # Transmit chain

    # Handle run time
    if run_time is not None:
        pi.wave_send_repeat(wid[-1])
        time.sleep(run_time)
        pi.wave_tx_stop()  # Stop waveform transmission
    else:
        # If no run_time specified, repeat the last waveform indefinitely
        pi.wave_send_repeat(wid[-1])

    # Clean up waveforms
    global last_wave_ids
    last_wave_ids = wid  # Store wave IDs globally to allow stopping later

def stop_motor():
    """Stop any running waveforms and clean up."""
    global last_wave_ids
    pi.wave_tx_stop()  # Stop any waveform transmission
    if last_wave_ids is not None:
        for wave_id in last_wave_ids:
            pi.wave_delete(wave_id)  # Clean up each waveform individually
    last_wave_ids = None

def align():
    """Adjust the motor to align the system based on the detected target offsets."""
    consecutive_aligned = 0
    while consecutive_aligned < REQ_CONSEC:
        if not target_offset_queue.empty():
            offset = target_offset_queue.get()

            # Calculate number of steps (proportional to the offset)
            steps = int(abs(offset) * X_OFFSET_CONV_FACTOR)

            #print(offset)
            # Calculate the step delay and direction based on offset
            if (offset > -20 and offset < 20) or steps < 10:
                consecutive_aligned += 1  # Increment if aligned
                continue
            else:
                consecutive_aligned = 0  # Reset if not aligned

            # Determine direction based on the sign of the offset
            direction = 1 if offset > 0 else 0

            move_motor(200, 200, 100, direction, steps / 200)

            time.sleep( steps / 200)

    # Stop the motor once aligned
    pi.write(STEP_PIN, 0)  # Ensuring no more steps are triggered
    print("camera aligned with target")

    # Align with datum
    move_motor(10, 200, 100, 0, DATUM_OFFSET/200)
    
    pi.write(STEP_PIN, 0)
    print("aligned")


# Initialize pigpio library instance and configure GPIO modes
pi = pigpio.pi()
if not pi.connected:
    print("Error connecting to pigpio daemon. Is the daemon running?")
pi.set_mode(STEP_PIN, pigpio.OUTPUT)
pi.wave_clear()

wave_ids = []  # Keep track of created wave IDs globally or in shared context

try:
    detector_thread = threading.Thread(target=detector)
    detector_thread.start()
    
    align_thread = threading.Thread(target=align)
    align_thread.start()
except KeyboardInterrupt:
    print("KeyboardInterrupt detected, stopping all threads.")
finally:
    detector_thread.join()
    #menu_thread.join()
    align_thread.join()