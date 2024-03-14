import os
import shutil
from tqdm import tqdm
root_path = r"D:\selectImg"
targetDir = r'dealImgV2'
resDir = r'res'
if not os.path.exists(os.path.join(root_path, resDir)):
    os.makedirs(os.path.join(root_path, resDir))
for dir in os.listdir(os.path.join(root_path, targetDir)):
    path = os.path.join(os.path.join(os.path.join(root_path, targetDir), dir), 'create')
    for file in tqdm(os.listdir(path)):
        shutil.copy(os.path.join(path, file), os.path.join(os.path.join(root_path, resDir), file))
