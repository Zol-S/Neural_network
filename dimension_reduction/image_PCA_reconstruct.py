import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA

img = mpimg.imread('wild.jpg')
'''print(img.shape)
print(img.shape)
plt.axis('off')
plt.imshow(img)'''

img_r = np.reshape(img, (225, 951))
ipca = PCA(64, svd_solver='randomized').fit(img_r)
img_c = ipca.transform(img_r)
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))

temp = ipca.inverse_transform(img_c)
temp = np.reshape(temp, (225, 317, 3))

plt.axis('off')
plt.imshow(temp.astype(np.uint8))
plt.show()
