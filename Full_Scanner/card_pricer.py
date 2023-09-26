import requests

class CardPricer:
    def __init__(self):
        self.base_url = "https://api.scryfall.com/cards/named"

    def get_card_price(self, card_name):
        # Make a request to the Scryfall API
        response = requests.get(self.base_url, params={"fuzzy": card_name})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract card price (assuming USD for simplicity)
            card_price = data.get("usd", "N/A")
            
            return card_price
        else:
            print(f"Error fetching price for {card_name}. Status Code: {response.status_code}")
            return "N/A"
