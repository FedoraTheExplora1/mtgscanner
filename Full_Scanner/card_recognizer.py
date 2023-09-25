import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
from collections import Counter
import json 
import requests

class CardRecognizer:
    def __init__(self):
        self.pytesseract = pytesseract 
        self.pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
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

    def extract_card_name(self, name_region):
        # Convert to grayscale
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Increase the resolution by resizing the name region
        enhanced_gray = cv2.resize(enhanced_gray, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
        
        # Binarize the image using adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Use pytesseract to recognize the text with enhanced configuration
        config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        raw_name = pytesseract.image_to_string(opened, config=config)
        
        # Post-process the OCR result
        lines = raw_name.strip().split('\n')
        card_name = max(lines, key=len) if lines else ""
        
        return card_name

    def predict_card(self, card_name):
        highest_similarity = 0
        best_match = "Unknown"
        
        # Compare the recognized card name with card names from the JSON file
        for known_name in self.card_names:
            similarity = fuzz.ratio(card_name.lower(), known_name.lower())
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = known_name
        
        return best_match, highest_similarity

    def process_card(self, card_image):  
        y_start = 10  # Adjust this value
        y_end = 50    # Adjust this value
        x_start = 20  # Adjust this value
        x_end = 300   # Adjust this value

        # Extract the region containing the card name
        name_region = card_image[y_start:y_end, x_start:x_end]

        # Extract the card name from the name region
        card_name = self.extract_card_name(name_region)

        # Predict the card based on the extracted name
        recognized_name, similarity = self.predict_card(card_name)

        # Increment the counter for the recognized card name
        self.card_name_counter[recognized_name] += 1

        # Check if the card name has been recognized the required number of times
        if self.card_name_counter[recognized_name] >= 10:
            if self.final_similarity < similarity:
                self.final_recognized_name = recognized_name
                self.final_similarity = similarity

                # Get the price of the final recognized card
                card_price = self.get_card_price(self.final_recognized_name)

                # Print the final recognized card name and price
                print("Final Recognized Card Name:", self.final_recognized_name)
                print("Final Similarity:", self.final_similarity)
                print("Card Price:", card_price)

    def get_card_price(self, card_name):
        try:
            response = requests.get(self.SCryfall_API_URL + card_name)
            if response.status_code == 200:
                data = response.json()
                price = data.get("prices", {}).get("usd", "Price not available")
                return price
            else:
                return "Price not available"
        except Exception as e:
            return str(e)
