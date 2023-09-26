import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from card_detector import CardDetector
from card_recognizer import CardRecognizer
from card_pricer import CardPricer
import threading

class GUI:
    def __init__(self, app):
        self.app = app

        # Initialize card detector, recognizer, and pricer
        self.card_detector = CardDetector()
        self.card_recognizer = CardRecognizer()
        self.card_pricer = CardPricer()

        # Create GUI components and layout
        self.create_gui()
        self.start_scanning()

    def create_gui(self):
        # Create labels for camera feed and name region
        self.camera_feed_label = ttk.Label(self.app)
        self.camera_feed_label.grid(row=0, column=0, padx=10, pady=10)

        self.name_region_label = ttk.Label(self.app)
        self.name_region_label.grid(row=0, column=1, padx=10, pady=10)

        # Create a button to start scanning
        self.scan_button = ttk.Button(self.app, text="Scan Card", command=self.start_scanning)
        self.scan_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        #Label to display card name
        self.recognized_name_label = ttk.Label(self.app, text="", font=("Arial", 16))
        self.recognized_name_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def update_gui(self, frame, name_region):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            self.camera_feed_label.imgtk = frame
            self.camera_feed_label.configure(image=frame)

        if name_region is not None:
            name_region = cv2.cvtColor(name_region, cv2.COLOR_BGR2RGB)
            name_region = Image.fromarray(name_region)
            name_region = ImageTk.PhotoImage(image=name_region)
            self.name_region_label.imgtk = name_region
            self.name_region_label.configure(image=name_region)

    def start_scanning(self):
    # Check if a thread is already running
        if hasattr(self, "scanning_thread") and self.scanning_thread.is_alive():
            return

        def scan_thread():
            try:
                # Get a frame from the card detector
                frame, name_region = self.card_detector.detect_card()

                # If a name region is detected, recognize the card's name
                if name_region is not None:
                    # Increase resolution and contrast of the name_region
                    name_region = cv2.resize(name_region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    name_region = cv2.convertScaleAbs(name_region, alpha=1.2, beta=50)

                    # Perform OCR on the enhanced name_region
                    card_name, confidence = self.card_recognizer.extract_card_name(name_region)

                    if card_name:
                        # Update the GUI label with the recognized card name and confidence score
                        self.recognized_name_label.config(text=f"Recognized Card: {card_name} (Confidence: {confidence:.2f})")

                        # Optionally, fetch the card price and display it
                        card_price = self.card_pricer.get_card_price(card_name)
                        self.recognized_name_label.config(text=f"Recognized Card: {card_name} (Confidence: {confidence:.2f}) - Price: {card_price}")

                # Update the GUI
                self.update_gui(frame, name_region)

            except Exception as e:
                print(f"Error: {e}")

            # Start the scanning thread
            self.scanning_thread = threading.Thread(target=scan_thread)
            self.scanning_thread.start()

            # Schedule the next scan with increased delay
            self.app.after(100, self.start_scanning)

# Create the main application window
app = tk.Tk()
app.title("MTG Card Scanner")

# Create an instance of the GUI
gui = GUI(app)

# Start the Tkinter main loop
app.mainloop()

# Release resources after closing the GUI
gui.card_detector.release_resources()
