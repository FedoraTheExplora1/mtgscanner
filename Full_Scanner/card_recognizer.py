import cv2
import pytesseract
from Levenshtein import distance
import json
import numpy as np

class CardRecognizer:
    def __init__(self):
        # Path to the Tesseract executable (update if needed)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Load the card names from the JSON file
        with open(r"C:\Users\conno\OneDrive\Documents\GitHub\mtgscanner\default-cards-20230924210754.json", "r", encoding="utf-8") as file:
            self.card_data = json.load(file)
            # Assuming the JSON structure is a list of dictionaries with a 'name' key for card names
            self.card_names_list = [card['name'] for card in self.card_data]

    def extract_card_name(self, name_region):
        # Convert to grayscale
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filtering
        filtered = cv2.bilateralFilter(gray, 5, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Denoising using morphological operations
        kernel = np.ones((2,2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # OCR with Tesseract
        ocr_result = pytesseract.image_to_string(opening, config='--psm 6').strip().replace('\n', ' ')
        
        # Post-processing: Match with dictionary
        card_name = min(self.card_names_list, key=lambda x: distance(ocr_result, x))
        
        return card_name
