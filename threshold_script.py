__author__ = 'ashleyng'

import cv2
from skimage.filter import threshold_adaptive




#
# def thresh(image, thresh_value, thresh_image):
#     retval, thresh_img = cv2.threshold(src=image,
#                                        thresh=thresh_value,
#                                        maxval=255,
#                                        type=cv2.THRESH_BINARY)
#     new_image = cv2.bitwise_or(thresh_img, thresh_image)
#     cv2.imshow("img", thresh_img)
#     cv2.imshow("img2", thresh_image)
#     cv2.imshow("window", new_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return new_image


def main():
    image = cv2.imread("images/test_images/im_5_small.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_image = threshold_adaptive(image, 250, offset=10)
    thresh_image = thresh_image.astype("uint8") * 255

    cv2.imwrite("images/threshold_images/im_5_small_threshold.png", thresh_image)



if __name__ == "__main__":
    main()