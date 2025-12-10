import os
import requests
import time


def download_files_from_github(path, model_name, max_retries=3):
    """Download model files from GitHub with validation and retry logic.

    Args:
        path: Destination path for the model file
        model_name: Name of the model to download
        max_retries: Maximum number of download attempts

    Returns:
        bool: True if download succeeded, False otherwise
    """
    if model_name not in ["u2net", "u2net_human_seg", "u2netp"]:
        print("Invalid model name, please use 'u2net' or 'u2net_human_seg' or 'u2netp'")
        return False

    print(f"downloading model [{model_name}] to {path} ...")

    urls = []
    if model_name == "u2net":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2aa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ad']
    elif model_name == "u2net_human_seg":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2haa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2had']
    elif model_name == 'u2netp':
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2netp.pth']

    try:
        os.makedirs(os.path.expanduser("~/.u2net"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return False

    # Expected file sizes (approximate, in bytes) for validation
    expected_sizes = {
        "u2net": 176000000,      # ~176 MB
        "u2net_human_seg": 176000000,  # ~176 MB
        "u2netp": 4500000,       # ~4.5 MB
    }

    for attempt in range(max_retries):
        try:
            # Remove any existing partial/corrupt file
            if os.path.exists(path):
                os.remove(path)

            with open(path, 'wb') as out_file:
                for i, url in enumerate(urls):
                    print(f'downloading part {i+1}/{len(urls)} of {model_name} (attempt {attempt+1}/{max_retries})')

                    response = requests.get(url, timeout=60)
                    response.raise_for_status()  # Raise error for bad status codes

                    out_file.write(response.content)
                    print(f'finished downloading part {i+1}/{len(urls)} of {model_name}')

            # Validate downloaded file size
            file_size = os.path.getsize(path)
            expected_size = expected_sizes.get(model_name, 0)

            # Allow 10% variance in file size
            if expected_size > 0:
                if file_size < expected_size * 0.5:
                    raise ValueError(f"Downloaded file is too small ({file_size} bytes, expected ~{expected_size} bytes). Download may be incomplete.")

            # Basic validation: file should not be empty
            if file_size < 1000:  # Less than 1KB is definitely wrong
                raise ValueError(f"Downloaded file is too small ({file_size} bytes). Download failed.")

            print(f"Successfully downloaded {model_name} ({file_size} bytes)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Network error downloading {model_name}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {model_name} after {max_retries} attempts.")
                # Clean up corrupt file
                if os.path.exists(path):
                    os.remove(path)
                return False

        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            # Clean up corrupt file
            if os.path.exists(path):
                os.remove(path)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {model_name} after {max_retries} attempts.")
                return False

    return False
