import os
import urllib.request

# URLs to download
files_to_download = {
    "yolov10x_best.pt": "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt",
    "input_sample.png": "https://raw.githubusercontent.com/moured/YOLOv10-Document-Layout-Analysis/main/images/input_sample.png"
}

def download_file(url, filename):
    """Download a file from a given URL and save it locally."""
    try:
        print(f"Checking if {filename} exists...")
        # Only download if the file doesn't already exist
        if not os.path.exists(filename):
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
        else:
            print(f"{filename} already exists. Skipping download.")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    for filename, url in files_to_download.items():
        download_file(url, filename)
