import requests

class CardPricer:
    def __init__(self):
        self.SCRYFALL_API_URL = "https://api.scryfall.com/cards/named?fuzzy="

    def get_card_price(self, card_name):
        try:
            response = requests.get(self.SCRYFALL_API_URL + card_name)
            if response.status_code == 200:
                data = response.json()
                price = data.get("prices", {}).get("usd", "Price not available")
                return price
            else:
                return "Price not available"
        except Exception as e:
            return str(e)
