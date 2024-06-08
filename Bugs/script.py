import os

folders = os.listdir('.')
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
for folder in folders:
    os.chdir(folder)
    os.system(f'python main.py > output.txt 2>&1')
    os.chdir(current_dir)