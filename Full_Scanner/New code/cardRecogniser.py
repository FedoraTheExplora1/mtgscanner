import cv2
import pytesseract
import tkinter as tk
from tkinter import Button, Canvas, Toplevel, filedialog
import requests
import json  # Import the json module
from PIL import Image, ImageTk

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variable to store the image
img = None

def preprocess_image(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Optionally resize the image
    enlarged = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return enlarged

def get_set_name(set_code):
    response = requests.get(f'https://api.scryfall.com/sets/{set_code}')
    if response.status_code == 200:
        set_data = response.json()
        return set_data.get('name', 'Unknown Set Name')
    else:
        return None

def get_card_name_roi(image, template):
    h, w = image.shape[:2]
    name_top, name_bottom = template['card_name'][0][1], template['card_name'][1][1]
    name_left, name_right = template['card_name'][0][0], template['card_name'][1][0]
    roi = image[name_top:name_bottom, name_left:name_right]
    return roi

def extract_set_code(roi):
    custom_config = r'--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(roi, config=custom_config)
    return text.strip()

def extract_card_name(roi):
    preprocessed = preprocess_image(roi)
    # Specify a whitelist of characters
    custom_config = r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 '
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text.strip()

def load_template():  # Define the load_template function
    with open('template.json', 'r') as f:
        return json.load(f)

def scan_card():
    global img
    template = load_template()  # Load the template
    if img is not None and template:
        set_code_roi = img[template['set_code'][0][1]:template['set_code'][1][1], template['set_code'][0][0]:template['set_code'][1][0]]
        set_code = extract_set_code(set_code_roi)
        set_name = get_set_name(set_code)
        card_name_roi = get_card_name_roi(img, template)
        card_name = extract_card_name(card_name_roi)
        
        # Draw bounding boxes
        cv2.rectangle(img, template['set_code'][0], template['set_code'][1], (0, 255, 0), 2)
        cv2.rectangle(img, template['card_name'][0], template['card_name'][1], (0, 0, 255), 2)
        
        # Display image with bounding boxes
        cv2.imshow('Image', img)
        
        if set_name:
            print(f'Set Code: {set_code}, Set Name: {set_name}, Card Name: {card_name}')
        else:
            print(f'Invalid Set Code: {set_code}')

def open_image():
    global img
    filepath = filedialog.askopenfilename(title='Select Image File',
                                          filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if filepath:
        img = cv2.imread(filepath)
        cv2.imshow('Image', img)



def main():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    locked_rect = None  # This will hold the locked-on rectangle

    while True:
        ret, frame = cap.read()  # Read frame from the webcam
        if not ret:
            print("Failed to grab frame.")
            break

        if locked_rect is None:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangle with the correct aspect ratio
            largest_rect = None
            largest_area = 0
            for contour in contours:
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                area = w * h
                aspect_ratio = w / h
                if largest_area < area < 200000 and 0.6 < aspect_ratio < 0.8:  # Adjust these values as needed
                    largest_area = area
                    largest_rect = rect

            # Optionally draw the largest rectangle on the frame for visualization
            if largest_rect is not None:
                x, y, w, h = largest_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        elif captured_image is not None:
            # Process the captured image with the overlay
            x, y, w, h = locked_rect
            overlay = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)
            overlay = cv2.resize(overlay, (w, h))
            for c in range(0, 3):
                captured_image[y:y+h, x:x+w, c] = captured_image[y:y+h, x:x+w, c] * (1 - overlay[:, :, 3] / 255.0) + overlay[:, :, c] * (overlay[:, :, 3] / 255.0)
            cv2.imshow('Place your card inside the template', captured_image)  # Show the captured image with overlay

        cv2.imshow('Place your card inside the template', frame)  # Show the frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('l') and largest_rect is not None:  # Press 'l' to lock onto the currently detected card shape and capture the image
            locked_rect = largest_rect
            captured_image = frame.copy()
        elif key == ord('c') and captured_image is not None:  # Press 'c' to process and scan the captured image
            upscaled = cv2.resize(captured_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # Upscale image
            scan_card(upscaled)

    cap.release()
    cv2.destroyAllWindows()

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open camera")

        self.canvas = Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_lock = Button(window, text="Lock & Capture", width=10, command=self.lock_and_capture)
        self.btn_lock.pack(anchor=tk.CENTER, expand=True)

        self.btn_scan = Button(window, text="Scan Card", width=10, command=self.scan_card_method)  # Update the command
        self.btn_scan.pack(anchor=tk.CENTER, expand=True)

        self.locked_rect = None
        self.captured_image = None

        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def lock_and_capture(self):
        ret, frame = self.vid.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangle with the correct aspect ratio
            largest_rect = None
            largest_area = 0
            for contour in contours:
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                area = w * h
                aspect_ratio = w / h
                if largest_area < area < 200000 and 0.6 < aspect_ratio < 0.8:  # Adjust these values as needed
                    largest_area = area
                    largest_rect = rect

            # Optionally draw the largest rectangle on the frame for visualization
            if largest_rect is not None:
                x, y, w, h = largest_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.captured_image = frame
            self.display_captured_image()

    def display_captured_image(self):
        top = Toplevel()
        top.title("Captured Image")
        canvas = Canvas(top, width=self.captured_image.shape[1], height=self.captured_image.shape[0])
        canvas.pack()
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        top.mainloop()
        
    def scan_card_method(self):  # Rename the method to avoid name conflict
        if self.captured_image is not None:
            upscaled = cv2.resize(self.captured_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # Upscale image
            scan_card(upscaled)  # Ensure scan_card function can handle cv2 images

    def scan_card(img):  # Add img as an argument
        template = load_template()  # Load the template
        if img is not None and template:
            set_code_roi = img[template['set_code'][0][1]:template['set_code'][1][1], template['set_code'][0][0]:template['set_code'][1][0]]
            set_code = extract_set_code(set_code_roi)
            set_name = get_set_name(set_code)
            card_name_roi = get_card_name_roi(img, template)
            card_name = extract_card_name(card_name_roi)
            
            # Draw bounding boxes
            cv2.rectangle(img, template['set_code'][0], template['set_code'][1], (0, 255, 0), 2)
            cv2.rectangle(img, template['card_name'][0], template['card_name'][1], (0, 0, 255), 2)
            
            # Display image with bounding boxes
            cv2.imshow('Image', img)
            
            if set_name:
                print(f'Set Code: {set_code}, Set Name: {set_name}, Card Name: {card_name}')
            else:
                print(f'Invalid Set Code: {set_code}')

# Create a window and pass it to the App class
App(tk.Tk(), "Magic Card Scanner")

if __name__ == "__main__":
    main()
