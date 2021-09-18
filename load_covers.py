import cv2
import os
import json

"""This is a script to take in the folder of covers, 
get the ORB features, store into a dictionary, and export 
a json file for later comparison processing"""

# Load file paths for covers
path = 'covers'
my_files = os.listdir(path)

# Setup ORB
orb = cv2.ORB_create(nfeatures=1250)

# Setup json
data = {}
data['cover_info'] = []

# Iterate through covers, run ORB, add features to dictionary
for cover in my_files:
    filepath = path + '/' + cover
    img_in = cv2.imread(filepath, 0)
    kp, desc = orb.detectAndCompute(img_in, None)
    new_dic = {'name': os.path.splitext(cover)[0],
               'path': filepath,
               'desc': desc.tolist(),
               'match': 0
               }
    data['cover_info'].append(new_dic)

# Export to file
with open('cover_info.json', 'w') as outfile:
    json.dump(data, outfile)

print(f"Complete! {len(data['cover_info'])} cover details captured")