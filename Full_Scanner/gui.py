import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from card_detector import CardDetector
from card_recognizer import CardRecognizer
from card_pricer import CardPricer

class GUI:
    def __init__(self, app):
        self.app = app

        # Initialize card detector, recognizer, and pricer
        self.card_detector = CardDetector()
        self.card_recognizer = CardRecognizer()
        self.card_pricer = CardPricer()

        # Create GUI components and layout
        self.create_gui()

    def create_gui(self):
        # Create labels for camera feed and name region
        self.camera_feed_label = ttk.Label(self.app)
        self.camera_feed_label.grid(row=0, column=0, padx=10, pady=10)

        self.name_region_label = ttk.Label(self.app)
        self.name_region_label.grid(row=0, column=1, padx=10, pady=10)

        # Create a button to start scanning
        self.scan_button = ttk.Button(self.app, text="Scan Card", command=self.start_scanning)
        self.scan_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

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
        # Get a frame from the card detector
        frame, name_region = self.card_detector.detect_card()

        if frame is not None:
            # Display the frame in the GUI
            self.update_gui(frame, name_region)

            # Detect the card and extract the name region
            card, name_region = self.card_detector.detect_card(frame)
            
            if card is not None:
                # Recognize the card's name
                card_name = self.card_recognizer.extract_card_name(name_region)

                if card_name:
                    # Get the card price
                    card_price = self.card_pricer.get_card_price(card_name)

                    # Display the recognized card name and price
                    print("Recognized Card Name:", card_name)
                    print("Card Price:", card_price)

                    # Update the GUI with the name region
                    self.update_gui(frame, name_region)

                    # Call the start_scanning method again after a short delay
                    self.app.after(10, self.start_scanning)


# Create the main application window
app = tk.Tk()
app.title("MTG Card Scanner")

# Create an instance of the GUI
gui = GUI(app)

# Start the Tkinter main loop
app.mainloop()
