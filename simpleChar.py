import cv2
import numpy as np
image = cv2.imread('GRPOLY_Dataset\GRPOLY-DB-MachinePrinted-B1\saripolos1.jpg')
height, width, channels = image.shape
print(height, width)
color = (0,0,255)
thickness = 2
points = [(633,1827), (633,1828), (635,1828), (635,1829), (636,1829), (636,1830), (637,1830), (637,1832), (638,1832), (638,1833), (639,1833), (639,1836), (640,1836), (640,1837), (643,1837), (643,1838), (646,1838), (646,1841), (647,1841), (647,1843), (648,1843), (648,1845), (649,1845), (649,1848), (650,1848), (650,1850), (651,1850), (651,1852), (652,1852), (652,1854), (653,1854), (653,1858), (654,1858), (643,1858), (643,1859), (642,1859), (642,1864), (641,1864), (641,1869), (539,1869), (539,1868), (534,1868), (534,1867), (531,1867), (531,1866), (436,1866), (436,1865), (435,1865), (435,1864), (434,1864), (434,1861), (433,1861), (433,1859), (432,1859), (432,1857), (431,1857), (431,1850), (430,1850), (431,1850), (431,1845), (432,1845), (432,1840), (433,1840), (433,1837), (434,1837), (434,1835), (435,1835), (435,1832), (436,1832), (436,1830), (437,1830), (437,1829), (437,1830), (438,1830), (437,1830), (437,1833), (555,1833), (555,1832), (581,1832), (581,1831), (606,1831), (606,1830), (631,1830), (631,1829), (632,1829), (632,1828), (633,1828)]

image = cv2.polylines(image, np.array([points]), True, color, thickness)
new_img = cv2.resize(image, (height//3, width//2))
cv2.imshow('Polygon', new_img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 