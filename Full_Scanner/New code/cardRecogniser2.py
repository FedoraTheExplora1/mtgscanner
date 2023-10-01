import cv2
import pytesseract
import tkinter as tk
from tkinter import Button, Canvas, Label, StringVar, Toplevel, Scale, HORIZONTAL
import requests
import json
from PIL import Image, ImageTk
import numpy as np

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    custom_config = r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 '
    text = pytesseract.image_to_string(preprocessed, config=custom_config)
    return text.strip()


def load_template():
    with open(r'C:\Users\conno\OneDrive\Documents\GitHub\mtgscanner\template.json', 'r', encoding='utf-8') as f:
        return json.load(f)


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.template = load_template()

        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open camera")

        self.canvas = Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(column=0, row=0, rowspan=4)

        self.btn_lock = Button(window, text="Lock & Capture", width=10, command=self.lock_and_capture)
        self.btn_lock.grid(column=1, row=0)

        self.btn_scan = Button(window, text="Scan Card", width=10, command=self.scan_card_method)
        self.btn_scan.grid(column=1, row=1)

        self.set_var = StringVar()
        self.set_label = Label(window, textvariable=self.set_var)
        self.set_label.grid(column=1, row=2)

        self.name_var = StringVar()
        self.name_label = Label(window, textvariable=self.name_var)
        self.name_label.grid(column=1, row=3)

        self.locked_rect = None
        self.captured_image = None
        
        self.brightness_slider = Scale(window, from_=-100, to_=100, orient=HORIZONTAL, resolution=1, label="Brightness")
        self.brightness_slider.set(0)  # Default brightness value
        self.brightness_slider.grid(column=1, row=4)

        self.contrast_slider = Scale(window, from_=0.0, to_=3.0, orient=HORIZONTAL, resolution=0.1, label="Contrast")
        self.contrast_slider.set(1.0)  # Default contrast value
        self.contrast_slider.grid(column=1, row=5)
        
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
        
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            adjusted_frame = self.adjust_brightness_and_contrast(frame)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)


    def lock_and_capture(self):
        ret, frame = self.vid.read()
        if ret:
            frame = self.auto_adjust_brightness_contrast(frame)
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
                src_corners = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype="float32")
                dst_dimensions = (635, 889)
                M = self.get_perspective_transform_matrix(src_corners, dst_dimensions)
                bird_eye_view = self.apply_perspective_transform(frame, M, dst_dimensions)
                overlay = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)
                result_image = self.overlay_template(bird_eye_view, overlay)
                self.captured_image = result_image
                self.display_captured_image()
                
    def adjust_brightness_and_contrast(self, image):
        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted



    def display_captured_image(self):
        top = Toplevel()
        top.title("Captured Image")
        canvas = Canvas(top, width=self.captured_image.shape[1], height=self.captured_image.shape[0])
        canvas.pack()
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        top.mainloop()

    def scan_card_method(self):
        if self.captured_image is not None:
            upscaled = cv2.resize(self.captured_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            self.scan_card(upscaled)

    def scan_card(self, img):
        template = load_template()
        if img is not None and template:
            set_code_roi = img[template['set_code'][0][1]:template['set_code'][1][1], template['set_code'][0][0]:template['set_code'][1][0]]
            set_code = extract_set_code(set_code_roi)
            set_name = get_set_name(set_code)
            card_name_roi = get_card_name_roi(img, template)
            card_name = extract_card_name(card_name_roi)
            cv2.rectangle(img, template['set_code'][0], template['set_code'][1], (0, 255, 0), 2)
            cv2.rectangle(img, template['card_name'][0], template['card_name'][1], (0, 0, 255), 2)
            cv2.imshow('Image', img)

            if set_name:
                print(f'Set Code: {set_code}, Set Name: {set_name}, Card Name: {card_name}')
                self.set_var.set(f'Set: {set_name}')
                self.name_var.set(f'Card Name: {card_name}')
            else:
                print(f'Invalid Set Code: {set_code}')

    def on_closing(self):
        self.vid.release()
        cv2.destroyAllWindows()
        self.window.destroy()

    def get_perspective_transform_matrix(self, src_corners, dst_dimensions):
        dst_corners = np.array([
            [0, 0],
            [dst_dimensions[0], 0],
            [dst_dimensions[0], dst_dimensions[1]],
            [0, dst_dimensions[1]]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        return M

    def apply_perspective_transform(self, image, M, dst_dimensions):
        warped = cv2.warpPerspective(image, M, (dst_dimensions[0], dst_dimensions[1]))
        return warped

    def overlay_template(self, image, overlay):
     # Resize the overlay to match the image dimensions
        overlay_resized = cv2.resize(overlay, (image.shape[1], image.shape[0]))

        # Now the overlay and image have matching dimensions, so we can blend them
        for c in range(0, 3):
            image[:, :, c] = image[:, :, c] * (1 - overlay_resized[:, :, 3] / 255.0) + overlay_resized[:, :, c] * (overlay_resized[:, :, 3] / 255.0)

        return image



# Create a window and pass it to the App class
App(tk.Tk(), "Magic Card Scanner")
