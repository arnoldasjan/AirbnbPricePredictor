import os

entries = os.listdir('images/')
for file in entries:
    new_name = file.split('?')[0]
    os.rename(f'images/{file}', f'images/{new_name}')