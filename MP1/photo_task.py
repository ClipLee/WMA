import cv2
# import os
import numpy as np

# absolute_path = os.path.join(os.getcwd(), 'Resources', 'ball.png');
i = cv2.imread(
    'C:/Users/klips/OneDrive/Dokumenty/PJATK/6 - Letni/WMA/MP1/resourses/ball.png')


def objectFinder(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 1. zamina na hsv

    low_red = np.array((0, 55, 55), np.uint8)
    high_red = np.array((5, 255, 255), np.uint8)

    mask_red = cv2.inRange(img_hsv, low_red, high_red)  # 2. maska kolorow

    kernel = np.ones((4, 4), np.uint8)  # 3. maska
    mask_no_noise = cv2.morphologyEx(
        mask_red, cv2.MORPH_OPEN, kernel)  # 4. poprawa obrazu
    mask_closed = cv2.morphologyEx(mask_no_noise, cv2.MORPH_CLOSE, kernel)

    # res1 = cv2.bitwise_and(img_hsv, img, mask=mask_no_noise)
    res = cv2.medianBlur(mask_closed, ksize=5)

    # result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_no_noise)

    # contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matrix = cv2.moments(res, 1)  # rozkald pikseli na pilce
    # M = cv2.moments(contours[0])

    # 4. oblciza srodek obiektu
    cx = int(matrix['m10'] / matrix['m00'])
    cy = int(matrix['m01'] / matrix['m00'])

    # 4.1 rysuje marker
    cv2.drawMarker(img, (int(cx), int(cy)), color=(0, 255, 0),  # rysowanie markera na pilce
                   markerType=cv2.MARKER_CROSS, thickness=2)
    return img


def main():
    cv2.imshow("result", objectFinder(i))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

7238