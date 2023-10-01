import cv2
import json
import numpy as np
from PIL import Image, ImageTK
import imagehash
import requests
from io import BytesIO
import tkinter as tk
from tkinter import messagebox

# Function to download and hash cards
def download_and_hash_cards():
    url = "https://api.scryfall.com/cards/search?q=set:alp"
    response = requests.get(url)
    cards = response.json()["data"]

    card_hashes = {}
    for card in cards:
        if 'image_uris' in card:
            image_url = card['image_uris']['normal']
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            hash = imagehash.average_hash(image)
            card_hashes[card['name']] = str(hash)

    with open('card_hashes.json', 'w') as f:
        json.dump(card_hashes, f)
# Load or download card hashes
try:
    with open('card_hashes.json', 'r') as f:
        card_hashes = json.load(f)
except FileNotFoundError:
    download_and_hash_cards()
    with open('card_hashes.json', 'r') as f:
        card_hashes = json.load(f)

# Convert card_hashes to imagehash.Hash objects
card_hashes = {name: imagehash.hex_to_hash(hash) for name, hash in card_hashes.items()}

# Function to compute the hash of a card image
def compute_card_hash(card):
    card = card.astype(np.float32)
    card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)  # Convert to RGB
    card = cv2.resize(card, (32, 32))  # Resize to 32x32 pixels
    card = card // 64 * 64 + 32  # Reduce colors
    card = cv2.cvtColor(card, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    return imagehash.average_hash(Image.fromarray(card.astype(np.uint8)))


# Function to find the closest matching card name for a given image hash
def find_closest_match(image_hash):
    min_distance = float('inf')
    closest_match = None
    
    for name, card_hash in card_hashes.items():
        distance = image_hash - card_hash
        if distance < min_distance:
            min_distance = distance
            closest_match = name
            
    return closest_match if min_distance < 10 else None  # Adjust the threshold as needed

# Function to order points for perspective transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Function to get the perspective transform of the card
def get_card_perspective(frame, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = np.array([c[0] for c in approx], dtype="float32")
        rect = order_points(pts)

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        # Apply CLAHE to normalize lighting conditions
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        warped = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

        return warped

    return None

# GUI for user interaction
def create_gui():
    root = tk.Tk()
    root.title("MTG Card Scanner")

    label = tk.Label(root, text="Press 's' to scan the card, 'q' to quit.")
    label.pack(padx=20, pady=20)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the scanner?"):
            cap.release()
            cv2.destroyAllWindows()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# Capture images from the webcam and identify cards
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()

# Create a GUI window with instructions
create_gui()

# Main loop for capturing and processing webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 5000:  # Adjust this value as needed
            card = get_card_perspective(frame, contour)
            if card is not None:
                cv2.imshow('Card', card)

    cv2.imshow('MTG Card Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to scan the card
        if card is not None and isinstance(card, np.ndarray):
            card_hash = compute_card_hash(card)
            match = find_closest_match(card_hash)

            if match:
                print(f"Card identified: {match}")
                messagebox.showinfo("Card Identified", f"Card Name: {match}")
            else:
                print("Card not recognized")
                messagebox.showwarning("Card Not Recognized", "The card could not be identified.")

    elif cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
