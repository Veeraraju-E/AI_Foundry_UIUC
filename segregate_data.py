import os
import pandas as pd
import shutil

# Load the Excel file
excel_path = 'C:/Users/Veeraraju_elluru/Downloads/get_metadata_v1_getExif_8.xlsx'  # Update with your actual file path
df = pd.read_excel(excel_path)

# Define the root path of images and the destination folders
image_root_path = 'C:/Users/Veeraraju_elluru/Downloads/data/DND-SB/images'  # Update with your actual images root path
output_root_path = 'C:/Users/Veeraraju_elluru/Downloads/AI Foundry'        # Output root path for the folders
folders = ['_PKCa', 'N_KCa', 'NP_Ca', 'NPKCa+M+S']

# Create output folders if they do not exist
for folder in folders:
    folder_path = os.path.join(output_root_path, folder)
    os.makedirs(folder_path, exist_ok=True)

# Iterate over the DataFrame and move images to the corresponding folders
for index, row in df.iterrows():
    image_name = row['img_names']
    image_path = os.path.join(image_root_path, image_name)
    
    if row['_PKCa'] == 1:
        shutil.copy(image_path, os.path.join(output_root_path, '_PKCa', image_name))
    if row['N_KCa'] == 1:
        shutil.copy(image_path, os.path.join(output_root_path, 'N_KCa', image_name))
    if row['NP_Ca'] == 1:
        shutil.copy(image_path, os.path.join(output_root_path, 'NP_Ca', image_name))
    if row['NPKCa+M+S'] == 1:
        shutil.copy(image_path, os.path.join(output_root_path, 'NPKCa+M+S', image_name))

print("Images have been successfully moved to their respective folders.")
