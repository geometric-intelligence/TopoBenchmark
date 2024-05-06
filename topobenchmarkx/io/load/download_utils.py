from urllib.parse import parse_qs, urlparse

import requests


# Function to extract file ID from Google Drive URL
def get_file_id_from_url(url):
    """
    Extracts the file ID from a Google Drive file URL.

    Args:
        url (str): The Google Drive file URL.

    Returns:
        str: The file ID extracted from the URL.

    Raises:
        ValueError: If the provided URL is not a valid Google Drive file URL.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:  # Case 1: URL format contains '?id='
        file_id = query_params["id"][0]
    elif "file/d/" in parsed_url.path:  # Case 2: URL format contains '/file/d/'
        file_id = parsed_url.path.split("/")[3]
    else:
        raise ValueError("The provided URL is not a valid Google Drive file URL.")
    return file_id


# Function to download file from Google Drive
def download_file_from_drive(
    file_link, path_to_save, dataset_name, file_format="tar.gz"
):
    """
    Downloads a file from a Google Drive link and saves it to the specified path.

    Args:
        file_link (str): The Google Drive link of the file to download.
        path_to_save (str): The path where the downloaded file will be saved.
        dataset_name (str): The name of the dataset.
        file_format (str, optional): The format of the downloaded file. Defaults to "tar.gz".

    Returns:
        None

    Raises:
        None
    """
    file_id = get_file_id_from_url(file_link)

    download_link = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_link)

    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")
