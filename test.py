import numpy as np
from cv2 import cv2
import matplotlib as matplotlib
import matplotlib.pyplot as plt

def access_image():

    # read
    img = cv2.imread('./tutorial/data/lecun.jpg')
    print('\ntype(img):', type(img))
    print('\nimg.shape:', img.shape)
    print('\nimg.dtype:', img.dtype)

    # display
    cv2.imshow('lecun', img); cv2.waitKey(10)

    # access
    print('\nimg[10][20] =', img[10][20])

    # resize
    h, w, _ = img.shape
    h_new, w_new = h // 2, w // 2
    print('\nresizing from (%d, %d) to (%d, %d)' % (h, w, h_new, w_new))

    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('lecun 1/2', img_resized); cv2.waitKey(0)

    # write
    cv2.imwrite('./tutorial/data/lecun_resized.jpg', img_resized)

    # destroy all windows
    cv2.destroyAllWindows()

def kmeans_color_clustering():

    X = np.random.randint(25,50,(25,2))
    Y = np.random.randint(60,85,(25,2))
    Z = np.vstack((X,Y))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # now separate the data, Note the flatten()
    A = Z[label.ravel()== 0]
    B = Z[label.ravel()== 1]

    # plot the data
    plt.scatter(A[:,0], A[:,1])
    plt.scatter(B[:,0], B[:,1], c = 'r')
    plt.scatter(center[:, 0], center[:, 1], s = 80, c = 'y', marker = 's')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()

def kmeans_color_quantization():

    # read
    img = cv2.imread('./tutorial/data/ryan.jpg')
    Z = img.reshape((-1, 3)).astype(np.float32)

    # number of clusters
    K = 8

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in range(K):
        this_data = Z[label.ravel() == idx]
        this_color = np.array([center[idx, 2] / 255, center[idx, 1] / 255, center[idx, 0] / 255])
        ax.scatter(this_data[::10, 0], this_data[::10, 1], this_data[::10,2], color = this_color)
    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')
    plt.show()

    # now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('input', img); cv2.waitKey(10)
    cv2.imshow('output', res2); cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./tutorial/data/ryan_quantized.jpg', res2)

if __name__ == "__main__":
    # access_image()
    # kmeans_color_clustering()
    kmeans_color_quantization()