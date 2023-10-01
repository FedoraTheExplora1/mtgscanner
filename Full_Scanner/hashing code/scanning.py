import cv2
import json
import numpy as np
from PIL import Image
import imagehash
import requests

# Function to download the card images and calculate their hashes
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

# Load card hashes from the JSON file
try:
    with open('card_hashes.json', 'r') as f:
        card_hashes = json.load(f)
except FileNotFoundError:
    download_and_hash_cards()
    with open('card_hashes.json', 'r') as f:
        card_hashes = json.load(f)

# Convert card_hashes values to imagehash.Hash objects for comparison
card_hashes = {name: imagehash.hex_to_hash(hash) for name, hash in card_hashes.items()}

# ... (rest of your code remains the same)

# Function to compute the hash of a card image
def compute_card_hash(card):
    card = card.astype(np.float32)
    card = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(card)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return imagehash.average_hash(Image.fromarray(card))

# ... (rest of your code remains the same)

while True:
    # ... (rest of your code remains the same)

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to scan the card
        if card is not None:
            card_hash = compute_card_hash(card)
            match = find_closest_match(card_hash)

            if match:
                print(f"Card identified: {match}")
            else:
                print("Card not recognized")

    elif cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
