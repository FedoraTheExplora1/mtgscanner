import cv2
import numpy as np

class CardDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            raise Exception("Could not open video capture. Check camera index.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame.")
        return frame

    def detect_card(self):
        frame = self.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            sorted_corners = sorted(approx, key=lambda x: x[0][1])
            top_points = sorted(sorted_corners[:2], key=lambda x: x[0][0])
            bottom_points = sorted(sorted_corners[2:], key=lambda x: x[0][0])

            if len(top_points) == 2:
                top_left = (top_points[0][0][0], top_points[0][0][1])
                top_right = (top_points[1][0][0], top_points[1][0][1])
            else:
                print("Error: Not enough top points detected.")
                return None, None

            if len(bottom_points) == 2:
                bottom_left = (bottom_points[0][0][0], bottom_points[0][0][1])
                bottom_right = (bottom_points[1][0][0], bottom_points[1][0][1])
            else:
                print("Error: Not enough bottom points detected.")
                return None, None

            # Calculate the height of the detected card in pixels
            card_height = bottom_left[1] - top_left[1]
            # Calculate the width of the detected card in pixels
            card_width = top_right[0] - top_left[0]

            # Calculate the start and end y-coordinates for the name region
            name_start_y = top_left[1]
            name_end_y = int(top_left[1] + (2/16) * card_height)  # Assuming the name region is about 2/16 inches tall

            # Calculate the end x-coordinate for the name region (leftmost 2/3 of the card width)
            name_end_x = top_left[0] + int(2/3 * card_width)

            # Extract the name region from the frame
            name_region = frame[name_start_y:name_end_y, top_left[0]:name_end_x]

            return frame, name_region

        return frame, None

    def release_resources(self):
        """Release the video capture object and destroy all OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()
