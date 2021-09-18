import cv2
import os
import json
import numpy as np
   
# Load in covers & covert back feature arrays
f = open('cover_info.json')
   
data = json.load(f)

for cover in data['cover_info']:
    cover['desc'] = np.array(cover['desc']).astype('uint8')

# Load ORB and BFM model
orb = cv2.ORB_create(nfeatures=1250)
bf = cv2.BFMatcher()

# Start capture
cap = cv2.VideoCapture(0)

while True: 
    comic_id = ""
    success, cap_img = cap.read()
    cap_copy = cap_img.copy()
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
    position = ((int) (cap_img.shape[1]/2 - 268/2), (int) (cap_img.shape[0]/2 - 36/2))

    kp, cap_desc = orb.detectAndCompute(cap_img, None)
    for cover in data['cover_info']:
        matches = bf.knnMatch(cap_desc, cover['desc'], k=2)
        good_matches = [m for m,n in matches if m.distance < .7*n.distance]
        if len(good_matches) > 12: 
            print(cover['name'])
            comic_id = cover['name']
    cap_copy = cv2.putText(cap_copy, comic_id, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (255,255,255), 7, cv2.LINE_AA)
    cv2.imshow('cap_img', cap_copy)        
    cv2.waitKey(1)