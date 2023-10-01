import requests
import sqlite3
import imagehash
from PIL import Image
from io import BytesIO
import concurrent.futures

def download_and_hash_card(card):
    try:
        image_url = card['image_uris']['normal']
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            hash = imagehash.average_hash(image)
            return card['name'], str(hash)
        print(f"Failed to download image for {card['name']}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while processing {card['name']}: {e}")
    return None, None

def download_and_hash_all_cards():
    conn = sqlite3.connect('card_hashes.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS card_hashes (
        name TEXT PRIMARY KEY,
        hash TEXT NOT NULL
    )
    ''')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Get all sets
    sets_url = "https://api.scryfall.com/sets"
    response = requests.get(sets_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve sets data. Status code: {response.status_code}")
        print(response.text)
        return

    sets = response.json()['data']

    for set in sets:
        url = f"https://api.scryfall.com/cards/search?q=set:{set['code']}"
        while url:
            print(f"Accessing URL: {url}")  
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    json_response = response.json()
                    cards = json_response['data']
                    url = json_response.get('next_page')

                    if not url:  
                        print("No more pages to retrieve.")
                        break

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(download_and_hash_card, card) for card in cards]
                        for future in concurrent.futures.as_completed(futures):
                            name, hash = future.result()
                            if name and hash:
                                cursor.execute("INSERT OR REPLACE INTO card_hashes (name, hash) VALUES (?, ?)", (name, hash))
                                print(f"Stored hash for {name}")

                    conn.commit()
                else:
                    print(f"Failed to retrieve cards data. Status code: {response.status_code}")
                    print(response.text)
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                break

    conn.close()

download_and_hash_all_cards()
