__author__ = 'ashleyng'

import cv2


def main():
    image = cv2.imread("images/abadi_condensed_light.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    retval, thresh = cv2.threshold(src=blur,
                                   thresh=127,
                                   maxval=225,
                                   type=cv2.THRESH_BINARY)
    contours, heirarchy = cv2.findContours(image=thresh,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    # draw contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img=image,
                      pt1=(x, y),
                      pt2=(x+w, y+h),
                      color=(0, 255, 0),
                      thickness=2)

    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()