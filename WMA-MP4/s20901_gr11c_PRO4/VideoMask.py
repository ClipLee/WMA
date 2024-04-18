import cv2
import numpy as np
from cv2 import imshow, rectangle


# classyfying faces to persons
jdepp_img1 = cv2.imread("images/jdepp/jdepp1.jpg")

mietczynski_img1 = cv2.imread("images/mietczynski/mietczynski1.jpg")
mietczynski_img2 = cv2.imread("images/mietczynski/mietczynski2.jpg")
mietczynski_img3 = cv2.imread("images/mietczynski/mietczynski3.jpg")

piszczek_img1 = cv2.imread("images/piszczek/piszczek1.jpg")
piszczek_img2 = cv2.imread("images/piszczek/piszczek2.jpg")

pudzian_img1 = cv2.imread("images/pudzian/pudzian1.jpg")
pudzian_img2 = cv2.imread("images/pudzian/pudzian2.jpg")

unknown_img1 = cv2.imread("images/unknown/unknown1.jpg")
unknown_img2 = cv2.imread("images/unknown/unknown2.jpg")
unknown_img3 = cv2.imread("images/unknown/unknown3.jpg")



def main():
    global swich_lib, fun, img
    # cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    scaling_factor = 0.5
    while True:
        ret, frame = cv2.imread("images/jdepp/jdepp1.jpg")
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = frame[y:y+h, x:x+w]
                img = cv2.resize(img, (200, 200),
                                 interpolation=cv2.INTER_LINEAR)
                cv2.imshow('twarz', img)

                if fun is not None:
                    fun()

        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()