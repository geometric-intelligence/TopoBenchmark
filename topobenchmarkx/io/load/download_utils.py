import requests
from urllib.parse import urlparse, parse_qs

def get_file_id_from_url(url):
    parsed_url = urlparse(url)
    if 'drive.google.com' not in parsed_url.netloc:
        raise ValueError("The provided URL is not a valid Google Drive link.")
    query_params = parse_qs(parsed_url.query)
    if 'id' not in query_params:
        raise ValueError("The provided URL does not contain a file ID.")
    return query_params['id'][0]

def download_file_from_drive(file_id, output_path):
    download_link = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_link)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")

def download_google_drive_datasets(file_link, path_to_save, dataset_name, file_format="tar.gz"):
    """
    Download the Cornell datasets from the provided link.

    Parameters
    ----------
    file_link : str
        Google drive link to download the file from.
    dataset_name : str
    """

    # Replace 'YOUR_LINK_HERE' with the link you want to download
    #file_link = "https://drive.google.com/open?id=1VA2P62awVYgluOIh1W4NZQQgkQCBk-Eu"

    # Extract file ID from the link
    file_id = get_file_id_from_url(file_link)

    # Download the file
    download_file_from_drive(file_id, output_path=f"{path_to_save}/{dataset_name}.{file_format}")
