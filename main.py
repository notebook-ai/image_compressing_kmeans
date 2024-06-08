import matplotlib as mpl
import matplotlib.pyplot as plt

image = mpl.image.imread('photo.jpg')

plt.imshow(image)

# # image size = 128 * 128 * 3 * 8 bit => 393216 bits
# # R 8bit 0-255
# # G 8bit 0-255
# # B 8bit 0-255

# # برای راحتی کار و اینکه مدل درک بهتری از مقادیر بازه 256 تایی داشته باشه 
# # ما آن را در بازه 0 تا 1 قرار می دهیم
image = image / 255
print(image.shape)

img = image.reshape(-1, 3)
print(img)

print(img.shape)




# k-means algorithm function
def k_means_all(x, centroids, K, max_iter, findClosestCentroid, computeCentroids):
    for i in range(max_iter):
        idx = findClosestCentroid(x, centroids)
        centroids = computeCentroids(x, idx, K)
    return idx, centroids

# # define some setting
K = 16 
max_iter = 10
initial_centroids = randomCentroids(img, K)
idx, centroids = k_means_all(img, initial_centroids, K, max_iter, findClosestCentroid, computeCentroids)


img_compressed = centroids[idx, :].reshape(image.shape)
plt.imshow(img_compressed)

# print(centroids)
# img_compressed[0,0,:]
# image[0,0,:]





