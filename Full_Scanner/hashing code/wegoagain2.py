import cv2
import imagehash
import sqlite3
import numpy as np
from PIL import Image, ImageTk
from tkinter import messagebox, StringVar, Tk, Label, Canvas, Button

# Function to connect to the SQLite database
def connect_to_db(db_name):
    """
    Connects to the SQLite database.
    Args:
        db_name: The name of the database.
    Returns:
        conn: The database connection object.
    """
    conn = sqlite3.connect(db_name)
    return conn

# Function to get all card hashes from the database
def get_all_card_hashes(conn):
    """
    Retrieves all card hashes from the database.
    Args:
        conn: The database connection object.
    Returns:
        card_hashes: A list of card hashes.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM card_hashes")
    return cursor.fetchall()

# Function to compute the hash of a card from its image
def compute_card_hash(card):
    """
    Computes the hash of a card from its image.
    Args:
        card: The card image.
    Returns:
        card_hash: The computed hash of the card.
    """
    card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    card = cv2.adaptiveThreshold(card, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    card = cv2.medianBlur(card, 3)
    return imagehash.average_hash(Image.fromarray(card))

# Function to remove glare from the frame
def remove_glare(frame):
    """
    Removes glare from the frame.
    Args:
        frame: The input frame.
    Returns:
        res: The frame with glare removed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res

# Function to find the card in the image
def find_card(img, thresh_c=10, kernel_size=(5, 5), size_thresh=10000):
    """
    Finds the card in the image.
    Args:
        img: The input image.
        thresh_c: Threshold constant for adaptive thresholding.
        kernel_size: Size of the kernel for morphological operations.
        size_thresh: Minimum contour area threshold.
    Returns:
        cnts_rect: A list of contours representing the card.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, thresh_c)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)  # Changed MORPH_OPEN to MORPH_CLOSE
    contours, _ = cv2.findContours(img_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_rect = [cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True) for cnt in contours if cv2.contourArea(cnt) >= size_thresh]
    return cnts_rect

# Function to find the closest match of the card in the database
def find_closest_match(conn, image_hash):
    """
    Finds the closest match of the card in the database.
    Args:
        conn: The database connection object.
        image_hash: The hash of the card image.
    Returns:
        closest_match: The closest match of the card.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Hash FROM card_hashes")
    card_hashes = cursor.fetchall()
    min_distance = float('inf')
    closest_match = None
    for name, hash_val in card_hashes:
        card_hash = imagehash.hex_to_hash(hash_val)
        distance = image_hash - card_hash
        if distance < min_distance:
            min_distance = distance
            closest_match = name
    return closest_match if min_distance < 10 else None

# Function to get the perspective of the card
def get_card_perspective(frame, contour):
    """
    Gets the perspective of the card.
    Args:
        frame: The input frame.
        contour: The contour representing the card.
    Returns:
        warped: The perspective-transformed card.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        rect = order_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        return warped
    else:
        print("Failed to get card perspective.")
        return None

# Function to order points for perspective transformation
def order_points(pts):
    """
    Orders the points for perspective transformation.
    Args:
        pts: The input points.
    Returns:
        rect: The ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# GUI class for the MTG Card Scanner
class MTGCardScannerGUI(Tk):
    def __init__(self, cap, conn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn = conn
        self.title("MTG Card Scanner")
        self.cap = cap
        self.status_var = StringVar()
        self.status_var.set("Press 'Capture Card' to take a still image.")
        self.label = Label(self, textvariable=self.status_var)
        self.label.pack(padx=20, pady=5)
        self.canvas = Canvas(self, width=640, height=480)
        self.canvas.pack(side="left", padx=10, pady=5)
        self.card_canvas = Canvas(self, width=640, height=480)
        self.card_canvas.pack(side="left", padx=10, pady=5)
        self.btn_capture = Button(self, text="Capture Card", command=self.capture_card)
        self.btn_capture.pack(padx=20, pady=5)
        self.btn_add = Button(self, text="Add to Inventory", command=self.add_to_inventory)
        self.btn_add.pack(padx=20, pady=5)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_webcam_feed()

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.cv2_image)
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            # Compute and display bird's eye view
            frame = remove_glare(frame)
            contours = find_card(frame)
            if contours:
                cv2.drawContours(self.cv2_image, contours, -1, (0, 255, 0), 2)
                contour = max(contours, key=cv2.contourArea)
                card = get_card_perspective(frame, contour)
                if card is not None:
                    self.card = card
                    card_image = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
                    card_image = Image.fromarray(card_image)
                    card_photo = ImageTk.PhotoImage(image=card_image)
                    self.card_canvas.create_image(0, 0, anchor="nw", image=card_photo)
                    self.card_canvas.image = card_photo
        self.after(1000, self.update_webcam_feed)  # Increased time interval to 1000 milliseconds

    def capture_card(self):
        ret, frame = self.cap.read()
        if ret:
            contours = find_card(frame)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                self.card = get_card_perspective(frame, contour)
                if self.card is not None:
                    self.status_var.set("Card captured. Press 'Scan Card' to identify.")
                    self.display_card()  # Display the captured card
                else:
                    self.status_var.set("Failed to get card perspective.")
                    messagebox.showwarning("Error", "Failed to get card perspective.")
            else:
                self.status_var.set("No card detected. Try again.")
        else:
            self.status_var.set("Failed to capture the image.")
            messagebox.showwarning("Error", "Failed to capture the image.")
        if hasattr(self, 'card') and self.card is not None:
            print("Scanning card...")
            self.status_var.set("Scanning card...")
            self.update()
            try:
                card_hash = compute_card_hash(self.card)
                match = find_closest_match(self.conn, card_hash)
                if match:
                    print(f"Card identified: {match}")
                    self.match = match  # Set the self.match attribute
                    self.status_var.set(f"Card identified: {match}")
                    messagebox.showinfo("Card Identified", f"Card Name: {match}")
                else:
                    print("Card not recognized.")
                    self.status_var.set("Card not recognized.")
                    messagebox.showwarning("Card Not Recognized", "The card could not be identified.")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.status_var.set(f"An error occurred: {e}")
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                cv2.destroyAllWindows()
        else:
            self.status_var.set("Failed to get card perspective.")  # Added this line to set status if card perspective is None
    def display_card(self):
        if self.card is not None:
            card_image = cv2.cvtColor(self.card, cv2.COLOR_BGR2RGB)
            card_image = Image.fromarray(card_image)
            card_photo = ImageTk.PhotoImage(image=card_image)
            self.card_canvas.create_image(0, 0, anchor="nw", image=card_photo)
            self.card_canvas.image = card_photo

    def add_to_inventory(self):
        if hasattr(self, 'match') and self.match:
            cursor = self.conn.cursor()
            cursor.execute("SELECT Quantity FROM inventory WHERE Name=?", (self.match,))
            row = cursor.fetchone()
            if row:
                new_quantity = row[0] + 1
                cursor.execute("UPDATE inventory SET Quantity=? WHERE Name=?", (new_quantity, self.match))
            else:
                # Here you should add the logic to get the card set and price
                card_set = "Example Set"  # Replace with actual logic
                price = "Example Price"  # Replace with actual logic
                cursor.execute("INSERT INTO inventory (Name, Set, Price, Quantity) VALUES (?, ?, ?, 1)",
                               (self.match, card_set, price))
            self.conn.commit()
            self.status_var.set(f"{self.match} added to inventory.")
        else:
            self.status_var.set("Identify a card first.")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the scanner?"):
            self.cap.release()
            self.conn.close()
            self.destroy()

# Connect to the SQLite database and create a table for card hashes
conn = connect_to_db('card_hashes.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS inventory
                  (Name TEXT, "Set" TEXT, Price TEXT, Quantity INTEGER)''')
conn.commit()

# Create an instance of the MTG Card Scanner GUI
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()
gui = MTGCardScannerGUI(cap, conn)
gui.mainloop()