import numpy as np
import cv2
from sklearn.cluster import KMeans

slika = cv2.imread('/home/student/Downloads/ruap/images/tomandjerry.jpg')
h, w = slika.shape[:2]

slika = cv2.cvtColor(slika, cv2.COLOR_BGR2LAB)
slika = slika.reshape((slika.shape[0] * slika.shape[1], 3))

n_k = int(input("Broj clustera:"))
k = KMeans(n_clusters = n_k)

segmenti = k.fit_predict(slika)
seg_slika = k.cluster_centers_.astype("uint8")[segmenti]

seg_slika = seg_slika.reshape((h, w, 3))
slika = slika.reshape((h, w, 3))

seg_slika = cv2.cvtColor(seg_slika, cv2.COLOR_LAB2BGR)
slika = cv2.cvtColor(slika, cv2.COLOR_LAB2BGR)

cv2.imshow('segmented.jpg', seg_slika)
cv2.waitKey(0)
cv2.destroyAllWindows()