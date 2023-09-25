import cv2
import numpy as np
import pytesseract
import requests
from fuzzywuzzy import fuzz
from collections import Counter
import json

class CardDetector:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.SCryfall_API_URL = "https://api.scryfall.com/cards/named?fuzzy="

        # Load card names from the JSON file
        with open(r'C:\Users\conno\OneDrive\Documents\Coding\mtgcardscanner\default-cards-20230924210754.json', encoding="utf-8") as json_file:
            self.card_data = json.load(json_file)
            self.card_names = [card["name"] for card in self.card_data]

        self.card_name_counter = Counter()
        self.final_recognized_name = None
        self.final_similarity = 0
        self.final_output_count = 0
        self.final_output_limit = 10
        self.scanning_enabled = False

    def detect_card(self):
        cap = cv2.VideoCapture(1)  # Use the camera with index 0, adjust as needed

        while self.scanning_enabled:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for contour in contours:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 4:
                        # Sort the corners of the card
                        sorted_corners = sorted(approx, key=lambda x: x[0][0])

                        # Check if at least four corners are detected
                        if len(sorted_corners) >= 4:
                            # Calculate the rounded corner radius (adjust as needed)
                            corner_radius = 30

                            # Create a list of rounded corner points
                            top_left, top_right, bottom_right, bottom_left = sorted_corners[:4]
                            rounded_corners = [
                                top_left,
                                (top_right[0] - corner_radius, top_right[1]),
                                top_right,
                                (bottom_right[0], bottom_right[1] - corner_radius),
                                bottom_right,
                                (bottom_left[0] + corner_radius, bottom_left[1]),
                                bottom_left,
                                (top_left[0], top_left[1] + corner_radius),
                            ]

                            # Draw the rounded polygon as the bounding box
                            cv2.fillPoly(frame, [np.array(rounded_corners, dtype=np.int32)], (0, 255, 0))

                            width = max(int(np.linalg.norm(bottom_right - bottom_left)), int(np.linalg.norm(top_right - top_left)))
                            height = max(int(np.linalg.norm(top_right - bottom_right)), int(np.linalg.norm(top_left - bottom_left)))

                            if width > 50 and height > 50:
                                max_dim = max(width, height)
                                
                                src_points = np.array([top_left[0], top_right[0], bottom_left[0], bottom_right[0]], dtype='float32')
                                dst_points = np.array([[0, 0], [max_dim, 0], [0, max_dim], [max_dim, max_dim]], dtype='float32')
                                
                                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                                warped = cv2.warpPerspective(frame, matrix, (max_dim, max_dim))
                                
                                name_start_y = 0
                                name_end_y = int(2/16 * max_dim)
                                name_end_x = int(max_dim / 2)
                                name_region = warped[name_start_y:name_end_y, 0:name_end_x]
                                
                                card_name = self.extract_card_name(name_region)
                                recognized_name, similarity = self.predict_card(card_name)

                                # Increment the counter for the recognized card name
                                self.card_name_counter[recognized_name] += 1

                                if self.final_output_count < self.final_output_limit:
                                    # Display the full card with bounding box
                                    cv2.imshow('Card Detection', frame)

                                    # Display the name region with recognized text
                                    cv2.imshow('Name Region', name_region)

                                    print("Recognized Card Name:", recognized_name)
                                    print("Similarity:", similarity)
                                    print("Final Recognized Card Name:", self.final_recognized_name)
                                    print("Final Output Count:", self.final_output_count)
                                    print("Card Name Counter:", dict(self.card_name_counter))

                                    self.final_output_count += 1

                                # Check if the card name has been recognized the required number of times
                                if self.card_name_counter[recognized_name] >= 10:
                                    if self.final_similarity < similarity:
                                        self.final_recognized_name = recognized_name
                                        self.final_similarity = similarity

                                        # Get the card data from Scryfall API
                                        card_data = self.get_card_data(recognized_name)

                                        if card_data:
                                            print("Final Recognized Card Name:", self.final_recognized_name)
                                            print("Final Card Data:", card_data)

                # Exit the loop when the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def extract_card_name(self, name_region):
        # Perform OCR (Optical Character Recognition) using pytesseract
        if name_region is not None:
            # Convert the name region to grayscale
            gray_name_region = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to make the text stand out
            _, thresholded_name = cv2.threshold(gray_name_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract to extract the text from the thresholded image
            extracted_name = pytesseract.image_to_string(thresholded_name, config='--psm 6')

            # Remove any leading/trailing whitespaces and convert to lowercase
            return extracted_name.strip().lower()

        return ""

    def predict_card(self, card_name):
        # Compare the extracted card name with known card names
        max_similarity = 0
        recognized_name = ""

        for known_name in self.card_names:
            similarity = fuzz.ratio(card_name, known_name)
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_name = known_name

        return recognized_name, max_similarity

    def get_card_data(self, card_name):
        # Use Scryfall API to fetch card data by name
        try:
            response = requests.get(self.SCryfall_API_URL + card_name)
            if response.status_code == 200:
                card_data = response.json()
                return card_data
        except Exception as e:
            print("Error fetching card data:", e)

        return None

    def start_scanning(self):
        self.scanning_enabled = True
        self.detect_card()

    def stop_scanning(self):
        self.scanning_enabled = False

if __name__ == "__main__":
    card_detector = CardDetector()
    card_detector.start_scanning()
