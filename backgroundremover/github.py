import os
import requests


def download_files_from_github(path, model_name):
    if model_name not in ["u2net", "u2net_human_seg", "u2netp"]:
        print("Invalid model name, please use 'u2net' or 'u2net_human_seg' or 'u2netp'")
        return
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
        return

    try:

        with open(path, 'wb') as out_file:
            for i, url in enumerate(urls):
                print(f'downloading part {i+1} of {model_name}')
                part_content = requests.get(url)
                out_file.write(part_content.content)
                print(f'finished downloading part {i+1} of {model_name}')
    except Exception as e:
        print(e)
