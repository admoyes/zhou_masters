from requests import get
import os
import zipfile

BASE_URL = "http://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/"


def download(url, target_path):
    """
        this will download a file from url and write it to target_path
    """
    print("[info] downloading", url)
    with open(target_path, "wb") as file:
        print("[info] opened", target_path)
        response = get(url, stream=True)
        if response.status_code == 200:
            for chunk in response:
                file.write(chunk)


def unzip(zip_path, output_dir):
    print("[info] unzipping", zip_path)
    # ensure the file exists
    assert os.path.isfile(zip_path)
    # assert will fail if file doesn't exist so
    # if we're past it, we're good. let's unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def make_dir(path):
    print("[info] trying to create", path)
    try:
        os.makedirs(path)
        print("[success] created", path)
    except:
        # os.makedirs will fail if the directory already exists.
        # 'pass' just tells python to carry on anyway
        print("[exception] when creating", path, "(this is probably fine)")
        pass


def download_dataset(zip_folder, data_folder):
    make_dir(zip_folder)
    make_dir(data_folder)

    print("[info] downloading dataset")
    # there are 15 zip files we need to download (1 for each class)
    for i in range(1, 16):
        url = BASE_URL + "leaf" + str(i) + ".zip"
        zip_path = os.path.join(zip_folder, "leaf" + str(i) + ".zip")
        if not os.path.isfile(zip_path):
            download(url, zip_path)
            unzip(zip_path, data_folder)
        else:
            print("[info]", zip_path, "already exists, skipping")
    print("[info] dataset downloaded")