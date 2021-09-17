import cv2
import os

path = 'covers'
my_files = os.listdir(path)
# cover1 = cv2.imread('train.png',0)
# cover2 = cv2.imread('XMen500kb.png',0)
# cover1_name = "X-MEN #1"
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher()

image_info = []
for cover in my_files:
    filepath = path + '/' + cover
    img_in = cv2.imread(filepath, 0)
    kp, desc = orb.detectAndCompute(img_in, None)
    new_dic = {'name': os.path.splitext(cover)[0],
               'path': filepath,
               'desc': desc,
               'match': 0
               }
    image_info.append(new_dic)

print(image_info)

cap = cv2.VideoCapture(0)

while True: 
    success, cap_img = cap.read()
    cap_copy = cap_img.copy()
    cap_img = cv2.cvtColor(cap_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cap_img', cap_copy)
    kp, cap_desc = orb.detectAndCompute(cap_img, None)
    for cover in image_info:
        matches = bf.knnMatch(cap_desc, cover['desc'], k=2)
        good_matches = [m for m,n in matches if m.distance < .7*n.distance]
        if len(good_matches) > 12: 
            print(cover['name'])
    cv2.waitKey(1)