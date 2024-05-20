import cv2
import numpy as np

cap = cv2.VideoCapture(0)

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

    # Draw contours
    contour_image = np.zeros_like(frame)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)  # Draw green contours

    cv2.imshow("frame", frame)
    cv2.imshow("Blue", blue_mask)
    cv2.imshow("gray", gray)
    cv2.imshow("median", median)
    cv2.imshow("contour_image", contour_image)  # Show the image with contours drawn on it

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
