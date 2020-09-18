import numpy as np
from cv2 import cv2
import matplotlib as matplotlib

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



# def kmeans_color_quantization():

#     # read
#     img = cv2.im

if __name__ == "__main__":
    # access_image()