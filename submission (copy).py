# Enter your code here
import numpy as np
import cv2
from scipy import spatial

rx, ry, rw, rh = 0, 0, 0, 0
tolerance_a, tolerance_b = 1.25, 2.5
sigma = 0.1

draw = False
frame_copy = None


def color_close(YCrCB_pixel, YCrCB_key, tola=0.1, tolb=0.5):
    dist = np.sqrt(
        ((YCrCB_pixel - YCrCB_key)**2)[1:].sum()
    )
    # dist = spatial.distance.cosine(YCrCB_key[1:], YCrCB_pixel[1:])   # return dist

    if dist < tola:
        return 0.0
    elif dist < tolb:
        return (dist - tola) / (tolb - tola)
    else:
        return 1.0


def draw_rectangle(event, x, y, flags, param):
    global rx, ry, rw, rh, draw
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        rx, ry, rw, rh = x, y, 0, 0

    if event == cv2.EVENT_LBUTTONUP:
        # frame_copy = frame.copy()
        rw = max(0, x - rx)
        rh = max(0, y - ry)
        if draw:
            cv2.rectangle(frame_copy, (rx, ry, rw, rh), (255, 0, 0))

    # if event == cv2.EVENT_LBUTTONUP:
    #     draw = False


def change_tolerance_a(val):
    global tolerance_a
    tolerance_a = val / 4


def change_tolerance_b(val):
    global tolerance_b
    tolerance_b = val / 4
    print(tolerance_a, tolerance_b)


def change_sigma(val):
    global sigma
    sigma = val / 100


def change_cast_level(val):
    pass


if __name__ == '__main__':
    cap = cv2.VideoCapture("greenscreen-demo.mp4")
    bg = cv2.imread("jwst.jpg")
    win_name = 'frame'
    cv2.namedWindow(win_name)
    cv2.createTrackbar("Tolerance slider (a)", win_name, 0, 100, change_tolerance_a)
    cv2.createTrackbar("Tolerance slider (b)", win_name, 0, 100, change_tolerance_b)
    cv2.createTrackbar("Softness", win_name, 0, 100, change_sigma)
    cv2.createTrackbar("Color cast", win_name, 0, 25, change_cast_level)

    cv2.setTrackbarPos("Tolerance slider (a)", win_name, int(tolerance_a * 4))
    cv2.setTrackbarPos("Tolerance slider (b)", win_name, int(tolerance_b * 4))

    cv2.setMouseCallback(win_name, draw_rectangle)


    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    processing_size = tuple([s//4 for s in size])
    bg = cv2.resize(bg, processing_size)
    # bg = cv2.cvtColor(bg, cv2.COLOR_BGR2YCrCb)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()
    frame_copy = frame.copy()
    while True:
        cv2.imshow(win_name, frame_copy)
        if cv2.waitKey(20) >= 0:
            break

    key_img = frame[ry:ry+rh, rx:rx+rw].mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    # print(key_img)
    ycrcb_key = cv2.cvtColor(key_img, cv2.COLOR_BGR2YCrCb).squeeze()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, processing_size)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_flat = ycrcb.reshape((-1, 3))

        mask = np.apply_along_axis(color_close, 1, ycrcb_flat, ycrcb_key, tola=tolerance_a, tolb=tolerance_b)

        # print(mask.min(), mask.max(), np.median(mask))
        mask = 1 - mask
        mask = np.expand_dims(mask.reshape(ycrcb.shape[:2]), 2)
        mask = np.repeat(mask, 3, axis=2)#.astype(np.uint8)
        cv2.imshow('mask', mask)

        mask = cv2.GaussianBlur(mask, (3, 3), sigma)
        out = np.maximum(frame - mask * key_img.squeeze(), 0) + mask * bg
        # out = ycrcb - mask * ycrcb_key + bg * mask
        out = out.astype(np.uint8)
        # out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

        cv2.imshow(win_name, out)
        if cv2.waitKey(1) == ord('q'):
            break
