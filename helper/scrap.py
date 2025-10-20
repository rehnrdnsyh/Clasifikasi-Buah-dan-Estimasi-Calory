import requests
from bs4 import BeautifulSoup
import re

def scrape_nutrition_data(food_name):
    """
    Scrape nutrition data from fatsecret.co.id
    
    Parameters:
        - food_name (str): Name of the fruit to search for
        
    Returns:
        - tuple: (nutrition_dict, default_volume)
    """
    # Normalize food name for URL
    food_name = food_name.replace(" ", "-").lower()
    
    # Handle special cases
    if food_name == 'ceri':
        food_name = 'ceri-manis'
    elif food_name == 'kiwi':
        food_name = 'buah-kiwi'
    
    url = f"https://www.fatsecret.co.id/kalori-gizi/umum/{food_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")

        # Find nutrition table
        table = soup.find("table", class_="generic spaced")
        volume_table = soup.find("table", class_='generic')
        
        # Label mapping for nutrition data
        label_map = {
            "Kal": "Kalori",
            "Lemak": "Lemak", 
            "Karb": "Karbohidrat",
            "Prot": "Protein"
        }
        
        result = {}
        default_volume = "100 gram"  # Default fallback

        # Extract nutrition data
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                for col in cols:
                    text = col.get_text(strip=True)
                    for prefix, label in label_map.items():
                        if text.startswith(prefix):
                            match = re.search(r'\d+[.,]?\d*', text)
                            if match:
                                value = match.group().replace(",", ".")
                                unit = " kcal" if label == "Kalori" else " g"
                                result[label] = value + unit
                            break

        # Extract default volume/portion
        if volume_table:
            selected_row = volume_table.find("tr", class_="selected")
            if selected_row:
                first_col = selected_row.find("td")
                if first_col:
                    default_volume = first_col.get_text(strip=True)

        return result, default_volume
        
    except requests.RequestException as e:
        print(f"Error fetching data for {food_name}: {e}")
        return {}, "100 gram"
    except Exception as e:
        print(f"Error parsing data for {food_name}: {e}")
        return {}, "100 gram"

# link porsi
def scrape_portion_links(food_name):
    """
    Scrape available portion options for a food item.
    
    Parameters:
        - food_name (str): Name of the food to search for
        
    Returns:
        - list: Dictionaries with portion text and URL query parameters
    """
    try:
        food_name = food_name.replace(" ", "-").lower()
        
        url = "https://www.fatsecret.co.id/kalori-gizi/umum/" + food_name
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Common portion types to look for
        label_map = [
            "100 gram",
            "1 mangkok", 
            "1 porsi",
            "1 tusuk",
            "1 gelas",
            "1 buah",
            "1 potong",
            "1 piring"
        ]

        tables = soup.find_all("table", class_="generic")
        portion_links_dict = {}

        for table in tables:
            links = table.find_all("a", href=True)
            for link in links:
                text = link.get_text(strip=True)
                if text in label_map and text not in portion_links_dict:
                    href = link["href"]
                    parsed = urlparse(href)
                    query = f"?{parsed.query}"
                    portion_links_dict[text] = query

        # Format the results for better structure
        portion_links = [
            {
                "text": key, 
                "url": value,
                "description": f"Porsi {key} untuk {food_name.replace('-', ' ')}"
            } 
            for key, value in portion_links_dict.items()
        ]
        
        return portion_links
    except Exception as e:
        print(f"Error scraping portion links: {e}")
        return []

# porsi nutrisi
def scrape_portion_nutrition(food_name):
    food_name = food_name.replace(" ", "-").lower()
    
    portion_links = scrape_portion_links(food_name)
    
    portion_nutrition = []
    for portion in portion_links:
        portion_text = portion["text"]
        portion_url = portion["url"]
        nutrition_data, volume = scrape_nutrition_data(food_name, portion_url)
        
        # Gabungkan data nutrisi dengan informasi porsi
        nutrition_data["porsi"] = portion_text
        nutrition_data["volume"] = volume
        portion_nutrition.append(nutrition_data)
        
    return portion_nutrition