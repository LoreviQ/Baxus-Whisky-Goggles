import pandas as pd

def fetch_images(images_directory: str, dataframe: pd.DataFrame):
    """
    Fetches images from the URLs in the provided dataframe and saves them in the specified directory.

    Args:
        images_directory (str): The directory to save the images.
        dataframe (pd.DataFrame): The dataframe containing image URLs and IDs.

    Raises:
        Exception: If an image cannot be fetched, an error message is printed.
    """
    import os
    import requests

    os.makedirs(images_directory, exist_ok=True)
    for _, row in dataframe.iterrows():
        image_url = row['image_url']
        image_id = row['id']
        image_path = os.path.join(images_directory, f"{image_id}.png")
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to fetch image {image_id}: {e}")