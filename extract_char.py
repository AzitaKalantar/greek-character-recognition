import cv2
import numpy as np
from bs4 import BeautifulSoup
import csv
from math import ceil
import pandas as pd
image = cv2.imread('GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B4\\blaxou4.jpg')
file = open("GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B4\\blaxou4.xml",
            "r", encoding="utf8")
# points = [(633,1827), (633,1828), (635,1828), (635,1829), (636,1829), (636,1830), (637,1830), (637,1832), (638,1832), (638,1833), (639,1833), (639,1836), (640,1836), (640,1837), (643,1837), (643,1838), (646,1838), (646,1841), (647,1841), (647,1843), (648,1843), (648,1845), (649,1845), (649,1848), (650,1848), (650,1850), (651,1850), (651,1852), (652,1852), (652,1854), (653,1854), (653,1858), (654,1858), (643,1858), (643,1859), (642,1859), (642,1864), (641,1864), (641,1869), (539,1869), (539,1868), (534,1868), (534,1867), (531,1867), (531,1866), (436,1866), (436,1865), (435,1865), (435,1864), (434,1864), (434,1861), (433,1861), (433,1859), (432,1859), (432,1857), (431,1857), (431,1850), (430,1850), (431,1850), (431,1845), (432,1845), (432,1840), (433,1840), (433,1837), (434,1837), (434,1835), (435,1835), (435,1832), (436,1832), (436,1830), (437,1830), (437,1829), (437,1830), (438,1830), (437,1830), (437,1833), (555,1833), (555,1832), (581,1832), (581,1831), (606,1831), (606,1830), (631,1830), (631,1829), (632,1829), (632,1828), (633,1828)]
contents = file.read()
soup = BeautifulSoup(contents, 'xml')
glyphs = soup.find_all('Glyph')
#color = (0, 0, 255)
#color2 = (0, 255, 0)
#thickness = 2
i = 0
data = []
lables_set = set()
#height, width, channels = image.shape
classes_train = pd.read_csv("training_data.csv")
for glyph in glyphs:

    points = []
    points_str = str(glyph.Coords['points']).split(" ")
    for point in points_str:
        points.append(tuple(map(int, point.split(','))))
    # print(points)
    # print(glyph.get_text())

    if len(points) != 4 or len(glyph.get_text()) < 4 or points[0][0] > points[2][0] or points[0][1] > points[2][1]:
        # image = cv2.polylines(image, np.array(
        # [points]), True, color2, thickness)
        continue
    '''

    hight = ceil((points[3][1]-points[0][1])/2)
    width = ceil((points[1][0]-points[0][0])/2)
    if hight > width:
        nimage = image[points[0][1]:points[3][1], points[0]
                       [0]+width-hight:points[0][0]+width+hight]
    else:
        nimage = image[points[0][1]+hight-width:points[0]
                       [1]+hight+width, points[0][0]:points[1][0]]

    '''

    nimage = image[points[0][1]-2:points[3]
                   [1]+2, points[0][0]-2:points[1][0]+2]
    try:
        i += 1
        cv2.imwrite(
            'C:\\Users\\parvi\\Desktop\\Project\\images\\final_test\{a}.jpg'.format(a=i), nimage)
        data.append(['{a}.jpg'.format(a=i), glyph.get_text()[3]])
        lables_set.add(glyph.get_text()[3])
    except:
        print(points)
        print(glyph.get_text())

#new_img = cv2.resize(image, (height//7*2, width//5*2))
#cv2.imshow('Polygon', new_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(data)
print(len(lables_set))

# print(points[0][1])
# height, width, channels = image.shape
# new_img = cv2.resize(image, (height//7*2, width//5*2))
# cv2.imshow('Polygon', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
new_data = []
labels_list = list(lables_set)
for row in data:
    a = classes_train[classes_train["char"] == row[1]]
    try:
        cl = int(a["class"])
        ch = str(a["char"])
    except:
        continue
    new_data.append([row[0], cl])

with open('annotations_file_test.csv', 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_data)
