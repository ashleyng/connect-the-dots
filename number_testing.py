__author__ = 'ashleyng'

import cv2
import numpy as np

SAMPLE_FILE_NAME = 'roman1'
TEST_FILE_NAME = 'roman_simple_test'


def main():

    # read in training data
    samples = np.loadtxt('data/' + SAMPLE_FILE_NAME + '_samples.data', np.float32)
    responses = np.loadtxt('data/' + SAMPLE_FILE_NAME + '_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.KNearest()
    model.train(samples, responses)

    # testing
    test_image = cv2.imread("images/test_images/" + TEST_FILE_NAME + ".png")
    output = np.zeros(test_image.shape, np.uint8)

    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    retval, thresh_image = cv2.threshold(src=blur,
                                         thresh=127,
                                         maxval=225,
                                         type=cv2.THRESH_BINARY)

    contours, heirarchy = cv2.findContours(image=thresh_image,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000 and area > 150:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 35:
                cv2.rectangle(img=test_image,
                              pt1=(x, y),
                              pt2=(x+w, y+h),
                              color=(0, 255, 0),
                              thickness=2)

                individual_num = thresh_image[y:y+h, x:x+w]
                individual_num_small = cv2.resize(individual_num, (10, 10))
                individual_num_small = individual_num_small.reshape((1, 100))
                individual_num_small = np.float32(individual_num_small)
                # find closest match number
                retval, results, neighbor_response, dists = model.find_nearest(individual_num_small, k=3)
                string = str(int((results[0][0])) - 48)
                print int(string)
                cv2.imshow("Test_img", test_image)
                cv2.imshow("img", individual_num)
                cv2.imshow("output", output)
                cv2.waitKey(0)
                if int(string) != 57:
                    cv2.putText(img=output,
                                text=string,
                                org=(x, y+h),
                                fontFace=0,
                                fontScale=1,
                                color=(0, 255, 0))
    cv2.imwrite('images/test_images/' + TEST_FILE_NAME + '_results.png', output)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()