__author__ = 'ashleyng'

import cv2
import numpy as np

FILE_NAME = "cambria_simple1"


def main():
    image = cv2.imread("images/samples/" + FILE_NAME + ".png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    retval, thresh = cv2.threshold(src=blur,
                                   thresh=127,
                                   maxval=225,
                                   type=cv2.THRESH_BINARY)
    contours, heirarchy = cv2.findContours(image=thresh,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    samples = np.empty((0, 100))
    key_responses = []
    # add valid keys
    # numbers
    keys = [i for i in range(48, 58)]
    # d for dot
    keys.append(100)
    # i for ignore
    keys.append(105)

    # draw contours
    for contour in contours:
        area = cv2.contourArea(contour)
        # upperbounds: for contour of whole image
        # lowerbounds: get rid of inner circles in 8, 9 etc. except for 0, b/c too big
        # and will get rid of dot
        if area < 1000 and area > 150:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img=image,
                          pt1=(x, y),
                          pt2=(x+w, y+h),
                          color=(0, 255, 0),
                          thickness=2)

            # show number
            individual_number = image[y:y+h, x:x+w]
            thresh_individual_image = thresh[y:y+h, x:x+w]
            individual_number_small = cv2.resize(thresh_individual_image, (10, 10))
            cv2.imshow("number", individual_number)
            key = cv2.waitKey(0)

            # take in keys
            if key == 27:
                break
            elif key in keys:
                key_responses.append(key)
                sample = individual_number_small.reshape((1, 100))
                samples = np.append(samples, sample, 0)

    # cv2.imshow("test", image)
    # cv2.waitKey(0)
    cv2.imwrite("images/contours/" + FILE_NAME + ".png", image)
    cv2.destroyAllWindows()

    # save files
    key_responses = np.array(key_responses, np.float32)
    key_responses = key_responses.reshape((key_responses.size, 1))
    np.savetxt('data/' + FILE_NAME + '_samples.data', samples)
    np.savetxt('data/' + FILE_NAME + '_responses.data', key_responses)


if __name__ == "__main__":
    main()