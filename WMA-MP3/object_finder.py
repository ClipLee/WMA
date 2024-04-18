import cv2 as cv
import numpy as np
import os

PATH_VIDEO = 'resources/video_source.MP4'
PATH_OBJECT = 'resources/foto/1.JPEG'

video = cv.VideoCapture(PATH_VIDEO)

image = cv.imread(PATH_OBJECT)
gimage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

model = cv.SIFT_create(nfeatures=256)

matcher = cv.BFMatcher(cv.NORM_L2)

keypoints1, descriptors1 = model.detectAndCompute(gimage, None)

def main():
    DIR = 'resources/foto'
    photos = []

    for entry in os.scandir(DIR):
        if entry.path.endswith('.jpeg' or '.jpg') and entry.is_file():
            try:
                while video.isOpened():
                    vinfo, frame = video.read()
                    if not vinfo:
                        break
                    gframe = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                    keypoints2, descriptors2 = model.detectAndCompute(gframe, None)
                    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

                    good = []
                    for m, n in matches:
                        if m.distance < 0.72 * n.distance:
                            good.append([m])
                    res_img = cv.drawMatchesKnn(entry.path, keypoints1, frame, keypoints2, good, outImg=None,
                                                flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    res_img = cv.resize(res_img, [1600, 800])
                    cv.imshow('ss', res_img)
                    if cv.waitKey(25) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(str(e))
video.release()
cv.destroyAllWindows()