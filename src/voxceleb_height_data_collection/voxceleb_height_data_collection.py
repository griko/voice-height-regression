import pandas as pd
import requests

def get_person_height_wikidata(name):
    # Search for the person on Wikidata
    wikidata_url = "https://www.wikidata.org/w/api.php"
    wikidata_params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json"
    }
    wikidata_response = requests.get(wikidata_url, params=wikidata_params).json()

    if wikidata_response["success"] and len(wikidata_response["search"]) > 0:
        entity_id = wikidata_response["search"][0]["id"]

        # Retrieve the person's height from Wikidata
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        entity_response = requests.get(entity_url).json()

        if "claims" in entity_response["entities"][entity_id]:
            claims = entity_response["entities"][entity_id]["claims"]
            if "P2048" in claims:  # P2048 is the property ID for height
                height = claims["P2048"][0]["mainsnak"]["datavalue"]["value"]["amount"]
                unit = claims["P2048"][0]["mainsnak"]["datavalue"]["value"]["unit"]
                return height, unit

def convert_to_cm(height, unit):
    height = float(height)
    if unit == 'http://www.wikidata.org/entity/Q11573':  # meters to cm
        return height * 100
    elif unit == 'http://www.wikidata.org/entity/Q174728':  # cm remains cm
        return height
    elif unit == 'http://www.wikidata.org/entity/Q3710':  # ft to cm
        return height * 30.48
    elif unit == 'http://www.wikidata.org/entity/Q218593':  # inches to cm
        return height * 2.54
    else:
        return None  # Handle unknown units gracefully

def main():
    # Load the final dataframe
    df = pd.read_csv('./final_dataframe_extended.csv')[['Name', 'gender_wiki', 'VoxCeleb_ID']].drop_duplicates()
    heights = {}
    units = {}

    for name in df['Name']:
        height = None
        unit = None
        try:
            height, unit = get_person_height_wikidata(name)
        except (KeyError, TypeError):
            print(f'Didn\'t find height of {name}')
        if height:
            heights[name] = height
            units[name] = unit
    
    units_df = pd.DataFrame(units.items(), columns=['Name', 'unit'])
    heights_df = pd.DataFrame(heights.items(), columns=['Name', 'height_raw'])
    merged_df = pd.merge(units_df, heights_df, on='Name')

    # units_df['unit'].unique()  # ['http://www.wikidata.org/entity/Q174728', 'http://www.wikidata.org/entity/Q11573', 'http://www.wikidata.org/entity/Q218593', 'http://www.wikidata.org/entity/Q3710']

    merged_df['height'] = merged_df.apply(lambda row: convert_to_cm(row['height_raw'], row['unit']), axis=1)

    merged_df.to_csv('./voxceleb_height.csv', index=False)

if __name__ == "__main__":
    main()
