import cv2

"""使用BackgroundSubtractorKNN实现运动检测的例子"""

bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
camera = cv2.VideoCapture("movie.mpg")
# camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    fgmask = bs.apply(frame)
    # 将非纯白色的所有像素都设为0
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                         iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("detection", frame)
    cv2.imshow("mog", fgmask)
    cv2.imshow("thresh", th)

    if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()
camera.release()
