import requests
import json

class CardPricer:
    def get_card_price(self, card_name):
        url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            card_data = response.json()
            return card_data.get('usd', 'Price not available')
        except requests.RequestException as e:
            return f"Network error: {e}"
        except json.JSONDecodeError:
            return "Error decoding the response."
