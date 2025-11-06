import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

competition = 'airbus-ship-detection'
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

print("Downloading dataset files...")
api.competition_download_files(competition, path=data_dir)

# The main download is a ZIP file: airbus-ship-detection.zip
main_zip = os.path.join(data_dir, f'{competition}.zip')

# Step 4: Extract the main ZIP
print("Extracting main ZIP...")
with zipfile.ZipFile(main_zip) as z:
    z.extractall(data_dir)

# Step 5: Extract inner ZIPs (train_v2.zip and test_v2.zip)
inner_zips = ['train_v2.zip', 'test_v2.zip']
for inner_zip in inner_zips:
    inner_path = os.path.join(data_dir, inner_zip)
    if os.path.exists(inner_path):
        print(f"Extracting {inner_zip}...")
        extract_dir = os.path.join(data_dir, inner_zip.replace('.zip', ''))
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(inner_path) as z:
            z.extractall(extract_dir)

# Cleanup zips to save space.
os.remove(main_zip)
for f in os.listdir(data_dir):
    if f.endswith('.zip'):
        os.remove(os.path.join(data_dir, f))

print("Download and extraction complete!")
