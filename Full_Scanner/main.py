import tkinter as tk
from tkinter import ttk
import cv2
from card_detector import CardDetector
from card_recognizer import CardRecognizer
from card_pricer import CardPricer
from gui import CardScannerGUI

def main():
    # Initialize the main application window
    app = tk.Tk()
    app.title("MTG Card Scanner")

    # Initialize the card detector, recognizer, and pricer
    detector = CardDetector()
    recognizer = CardRecognizer()
    pricer = CardPricer()

    # Initialize the GUI
    gui = CardScannerGUI(app)

    # Create a video capture object
    cap = cv2.VideoCapture(2)  # Use your desired camera index

    def update_gui():
        ret, frame = cap.read()
        if ret:
            # Display the camera feed in the GUI
            gui.update_camera_feed(frame)

            # Perform card detection and recognition
            card_image, card_name = detector.detect_and_extract_card(frame)
            recognized_name, similarity = recognizer.predict_card(card_name)

            # Update the GUI with recognized card information
            gui.update_card_info(recognized_name, similarity)

            # Check if the card name has been recognized the required number of times
            if recognizer.card_name_counter[recognized_name] >= 10:
                if recognizer.final_similarity < similarity:
                    recognizer.final_recognized_name = recognized_name
                    recognizer.final_similarity = similarity

                    # Get the price of the final recognized card
                    card_price = pricer.get_card_price(recognized_name)

                    # Update the GUI with the final recognized card name and price
                    gui.update_final_card_info(recognized_name, recognizer.final_similarity, card_price)
                    
        app.after(10, update_gui)

    # Start updating the GUI and performing recognition
    update_gui()

    # Start the Tkinter main loop
    app.mainloop()

if __name__ == "__main__":
    main()
