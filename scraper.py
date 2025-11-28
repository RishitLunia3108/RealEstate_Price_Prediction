import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
 
def get_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
 
def extract_bhk(text):
    if not text: return "N/A"
    match = re.search(r'(\d+)\s*BHK', text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} BHK"
    return "N/A"
 
def extract_bathrooms(text):
    if not text: return "N/A"
    match = re.search(r'(\d+)\s*Bath', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"
 
def extract_furnishing(text):
    if not text: return "N/A"
    text = text.lower()
    if 'unfurnished' in text or 'un-furnished' in text:
        return 'Unfurnished'
    elif 'semi-furnished' in text or 'semifurnished' in text:
        return 'Semi-Furnished'
    elif 'fully furnished' in text or 'furnished' in text:
        return 'Furnished'
    return "N/A"
 
def extract_seller_type(text):
    if not text: return "N/A"
    text = text.lower()
    if 'owner' in text:
        return 'Owner'
    elif 'agent' in text or 'dealer' in text or 'broker' in text:
        return 'Agent'
    elif 'builder' in text or 'developer' in text:
        return 'Builder'
    return "N/A" # Default or unknown
 
def scrape_housing_locality(city_name, locality_name, max_pages=50):
    """Scrape Housing.com for a specific locality with deep pagination"""
    print(f"  Scraping {locality_name}...")
    all_properties = []
   
    for page in range(1, max_pages + 1):
        url = f"https://housing.com/in/buy/{city_name.lower()}/{locality_name.lower()}?page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=15)
           
            if response.status_code == 406:
                print(f"    Page {page}: Rate limited (406), stopping this locality.")
                break
            elif response.status_code != 200:
                print(f"    Page {page}: Status {response.status_code}, stopping.")
                break
           
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('article')
            if not listings:
                listings = soup.find_all('div', class_=lambda x: x and 'article' in x)
           
            if not listings:
                print(f"    Page {page}: No listings, stopping.")
                break
           
            if page % 5 == 0:
                print(f"    Page {page}: Found {len(listings)} listings")
           
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                   
                    title_tag = listing.find('h2') or listing.find('h3') or listing.find('div', {'data-q': 'title'})
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                   
                    price_tag = listing.find('div', {'data-q': 'price'}) or listing.find(lambda tag: tag.name == 'div' and '₹' in tag.text)
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                   
                    area_tag = listing.find('div', {'data-q': 'area'})
                    data['Area'] = area_tag.get_text(strip=True) if area_tag else "N/A"
                   
                    # New Fields Extraction
                    data['BHK'] = extract_bhk(data['Property Title'])
                    if data['BHK'] == "N/A":
                         data['BHK'] = extract_bhk(raw_text)
                         
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                   
                    # Seller Type - often in a specific tag or just text
                    seller_tag = listing.find('div', class_='seller-name') # Hypothetical class
                    if seller_tag:
                         data['Seller Type'] = extract_seller_type(seller_tag.get_text())
                    else:
                         data['Seller Type'] = extract_seller_type(raw_text)
 
                    data['Locality'] = locality_name
                    data['City'] = city_name
                    data['Source'] = 'Housing.com'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except:
                    continue
           
            # Vary delay to avoid detection
            time.sleep(random.uniform(0.5, 1.5))
           
        except Exception as e:
            print(f"    Page {page}: Error - {str(e)[:50]}, stopping.")
            break
   
    print(f"  [+] {locality_name}: Collected {len(all_properties)} records")
    return all_properties
 
def scrape_magicbricks(city_name, max_pages=50):
    """Scrape MagicBricks for a specific city"""
    print(f"  Scraping MagicBricks for {city_name}...")
    all_properties = []
   
    base_url = "https://www.magicbricks.com/property-for-sale/residential-real-estate"
   
    for page in range(1, max_pages + 1):
        # Construct URL with pagination
        url = f"{base_url}?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName={city_name}&page={page}"
       
        try:
            response = requests.get(url, headers=get_headers(), timeout=15)
           
            if response.status_code != 200:
                print(f"    Page {page}: Status {response.status_code}, stopping.")
                break
           
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('div', class_='mb-srp__card')
           
            if not listings:
                print(f"    Page {page}: No listings found, stopping.")
                break
           
            if page % 5 == 0:
                print(f"    Page {page}: Found {len(listings)} listings")
           
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                   
                    # Title
                    title_tag = listing.find('h2', class_='mb-srp__card--title')
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                   
                    # Price
                    price_tag = listing.find('div', class_='mb-srp__card__price--amount')
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                   
                    # Area
                    area_tag = listing.find('div', class_='mb-srp__card__summary--value')
                    data['Area'] = area_tag.get_text(strip=True) if area_tag else "N/A"
                   
                    # Location
                    data['Locality'] = "Ahmedabad" # Default
                    if title_tag:
                         title_text = title_tag.get_text(strip=True)
                         if " in " in title_text:
                             data['Locality'] = title_text.split(" in ")[-1].strip()
                   
                    # New Fields Extraction
                    data['BHK'] = extract_bhk(data['Property Title'])
                   
                    # MagicBricks often has summary items
                    summary_items = listing.find_all('div', class_='mb-srp__card__summary--item')
                    data['Bathrooms'] = "N/A"
                    data['Furnishing'] = "N/A"
                   
                    for item in summary_items:
                        label = item.find('div', class_='mb-srp__card__summary--label')
                        value = item.find('div', class_='mb-srp__card__summary--value')
                        if label and value:
                            lbl = label.get_text(strip=True).lower()
                            val = value.get_text(strip=True)
                            if 'bath' in lbl:
                                data['Bathrooms'] = val
                            elif 'furnishing' in lbl:
                                data['Furnishing'] = val
                   
                    if data['Bathrooms'] == "N/A":
                        data['Bathrooms'] = extract_bathrooms(raw_text)
                    if data['Furnishing'] == "N/A":
                        data['Furnishing'] = extract_furnishing(raw_text)
                       
                    # Seller Type
                    advertiser = listing.find('div', class_='mb-srp__card__ads--name')
                    if advertiser:
                        data['Seller Type'] = extract_seller_type(advertiser.get_text(strip=True))
                    else:
                        data['Seller Type'] = extract_seller_type(raw_text)
 
                    data['City'] = city_name
                    data['Source'] = 'MagicBricks'
                    data['Raw_Details'] = raw_text
                   
                    all_properties.append(data)
                except Exception as e:
                    continue
           
            # Vary delay
            time.sleep(random.uniform(1.0, 2.0))
           
        except Exception as e:
            print(f"    Page {page}: Error - {str(e)[:50]}, stopping.")
            break
           
    print(f"  [+] MagicBricks {city_name}: Collected {len(all_properties)} records")
    return all_properties
 
def main():
    # Target only Ahmedabad as requested
    target_city = 'Ahmedabad'
   
    print("\n" + "="*80)
    print(f"AHMEDABAD REAL ESTATE DATA COLLECTION")
    print("="*80)
    print(f"Target City: {target_city}")
    print(f"Sources: Housing.com, MagicBricks")
    print("="*80 + "\n")
   
    all_data = []
   
    # 1. Scrape Housing.com
    print(f"\n--- Source 1: Housing.com ---")
    housing_data = scrape_housing_locality(target_city, target_city, max_pages=150)
    all_data.extend(housing_data)
   
    # 2. Scrape MagicBricks
    print(f"\n--- Source 2: MagicBricks ---")
    mb_data = scrape_magicbricks(target_city, max_pages=150)
    all_data.extend(mb_data)
 
    if all_data:
        df_final = pd.DataFrame(all_data)
       
        # Reorder columns for better readability
        cols = ['Property Title', 'Price', 'Area', 'BHK', 'Bathrooms', 'Furnishing', 'Seller Type', 'Locality', 'City', 'Source', 'Raw_Details']
        # Ensure all cols exist
        for col in cols:
            if col not in df_final.columns:
                df_final[col] = "N/A"
       
        df_final = df_final[cols]
       
        # Remove duplicates
        df_final_unique = df_final.drop_duplicates(subset=['Property Title', 'Price', 'Area'], keep='first')
       
        # Create data directory if it doesn't exist
        import os
        os.makedirs('data', exist_ok=True)
        
        filename = 'data/ahmedabad_real_estate_data.csv'
        df_final_unique.to_csv(filename, index=False)
       
        print("\n" + "="*80)
        print("SCRAPING COMPLETE!")
        print("="*80)
        print(f"Total Records Scraped: {len(df_final):,}")
        print(f"Unique Records: {len(df_final_unique):,}")
        print(f"Housing.com: {len(housing_data):,}")
        print(f"MagicBricks: {len(mb_data):,}")
        print(f"Output File: {filename}")
        print("="*80)
    else:
        print("\n⚠ WARNING: No data scraped.")
 
if __name__ == "__main__":
    main()