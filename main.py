# افزودن کتابخانه های مورد نیاز
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# خواندن فایل تصویر
image = mpl.image.imread('photo.png')

# نشان دادن عکس 
# plt.imshow(image)

#توضیحات اضافی درک بهتر 
# # image size = 128 * 128 * 3 * 8 bit => 393216 bits
# # R 8bit 0-255
# # G 8bit 0-255
# # B 8bit 0-255


# # برای راحتی کار و اینکه مدل درک بهتری از مقادیر بازه 256 تایی داشته باشه 
# # ما آن را در بازه 0 تا 1 قرار می دهیم
img = image / 255
# print(image.shape)
# img = image.reshape(-1, 3)
# print(img)
# print(img.shape)



# K-means تابع الگوریتم
def K_means_all(X, centroids, K, max_iter, findClosestCentroid, computeCentroids):
  for i in range(max_iter):
    idx = findClosestCentroid(X, centroids)
    centroids = computeCentroids(X, idx, K)
  return idx, centroids 


# Choose random initial centroids
def randomCentroids(X, K):
  n = X.shape
  centroids = np.zeros((K, n))
  rand_idx = np.random.permutation(m)
#   print(rand_idx)
  centroids = X[rand_idx[0:K], :]
#   print(centroids)
  return centroids


# closes centroid to each data
def findClosestCentroid(X, centroids): 
  m = X.shape[0]
  K = centroids.shape[0]
  idx = np.zeros(m, dtype=int)
  #print(m)
  for i in range(m): 
    distance_array = np.sqrt(np.sum(np.square(X[i] - centroids), axis=1))
    idx[i] = np.argmin(distance_array)
  return idx


def computeCentroids(X, idx, K):
  n = X.shape
  centroids = np.zeros((K, n))
  for k in range(K):
    centroids[k] = np.mean(X[idx == k], axis=0)
    #centroid[0] = np.mean(X[idx == 0], axis=0)
  return centroids

# K : define number of colors 
# تنظیمات تعداد خوشه بندی 
# که می تواند اعدادی مانند 
# 4, 8, 16, 32, 64, 128, 256 
# را دریافت کند 
K=4

max_iter = 10
initial_centroids = randomCentroids(img, K)
idx, centroids = K_means_all(img, initial_centroids, K, max_iter, findClosestCentroid, computeCentroids)

img_compressed = centroids[idx, :].reshape(image.shape)

# نشان دادن عکس بعد از عملیات فشرده سازی
plt.imshow(img_compressed)




