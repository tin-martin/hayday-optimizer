import json
import re
import requests
from bs4 import BeautifulSoup

def parse_time_to_minutes(time_str):
    if time_str == 'Instant':
        return 0
    if time_str == 'Instantor 20 h':
        return 0

    time_str = time_str.lower().strip()
    total = 0

    if '★★★' in time_str:
        time_str = time_str.split('★★★')[0]

    if 'd' in time_str:
        parts = time_str.split('d')
        days = int(parts[0].strip())
        total += days * 24 * 60
        time_str = parts[1] if len(parts) > 1 else ''

    if 'h' in time_str:
        parts = time_str.split('h')
        hours = int(parts[0].strip())
        total += hours * 60
        time_str = parts[1] if len(parts) > 1 else ''

    if 'min' in time_str:
        min_part = time_str.split('min')[0].strip()
        if min_part:
            total += int(min_part)

    return total

def parse_ingredient(ingredient):
    # Match pattern like "Name (2)" or just "Name"
    match = re.match(r'^(.*?)\s*\((\d+)\)$', ingredient.strip())
    if match:
        name = match.group(1).strip()
        qty = int(match.group(2))
    else:
        name = ingredient.strip()
        qty = 1
    return name, qty

def scrape_goods_list():
    url = "https://hayday.fandom.com/wiki/Goods_List"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})

    goods_data = {}

    for row in table.find_all('tr')[1:]:
        cols = row.find_all(['td', 'th'])
        if len(cols) < 7: # Changed from 6 to 7 to account for the 'Source' column
            continue

        name = cols[0].get_text(strip=True)
        time_text = cols[3].get_text(strip=True)
        xp_text = cols[4].get_text(strip=True)
        needs_cell = cols[5]
        machine_text = cols[6].get_text(strip=True) # Extract machine text from the 6th column (index 5)

        raw_ingredients = " ".join(needs_cell.stripped_strings)

        # Regex to match each ingredient line
        pattern = r'(.*?)\s*\((\d+)\)'
        matches = re.findall(pattern, raw_ingredients)

        # Handle N/A case
        if any('n/a' in s.lower() for s in raw_ingredients):
            ingredients_dict = {}
        else:
            ingredients_dict = {}
            for label, count in matches:
                if label and label.lower() != 'n/a':
                    ingredients_dict[label] = count

        ingredients_dict = {
            key.strip(): int(value)
            for key, value in ingredients_dict.items()
        }
        time_min = parse_time_to_minutes(time_text)

        try:
            xp = int(xp_text)
        except ValueError:
            xp = 0

        # Clean up machine text, removing parenthetical descriptions like "(1st crop)"
        machine = re.sub(r'\s*\(.*\)', '', machine_text).strip()


        goods_data[name] = {
            "ingredients": ingredients_dict,
            "time": time_min,
            "xp": xp,
            "machine": machine # Add the machine variable
        }

    with open("hayday_goods.json", "w") as f:
        json.dump(goods_data, f, indent=2)

    print(f"✅ Saved {len(goods_data)} goods to hayday_goods.json")

if __name__ == "__main__":
    scrape_goods_list()