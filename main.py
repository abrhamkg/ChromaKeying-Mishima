import argparse

import cv2
import numpy as np

import mishima

DEBUG = False
# Variables that define the selected rectangle
rx, ry, rw, rh = 0, 0, 0, 0

# Variable to hold the values of constants for Vlahos alpha matting
tolerance_a, tolerance_b = 0.1, 0.5

# The standard deviation of Gaussian blurring used for alpha matte smoothing
sigma = 0.88

# A constant controlling the degree to which we correct for green spill
cast_factor = 0.5

#
REGION_SELECTION_MODE = True
LAST_REGION = list()  # A list that holds the last drawn rectangle
BG_REGIONS = list()  # A list that holds all rectangular regions assigned as definite background by user
FG_REGIONS = list()  # A list that holds all rectangular regions assigned as definite foreground by user


def select_regions(event, x, y, flags, param):
    global rx, ry, rw, rh, draw, REGION_SELECTION_MODE
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        rx, ry, rw, rh = x, y, 0, 0

    if event == cv2.EVENT_LBUTTONUP and REGION_SELECTION_MODE:
        # frame_copy = frame.copy()
        rw = max(0, x - rx)
        rh = max(0, y - ry)
        REGION_SELECTION_MODE = False

        rect = (rx, ry, rw, rh)
        LAST_REGION.append(rect)
        if draw:
            cv2.rectangle(frame_copy, rect, (255, 0, 0))


def change_tolerance_a(val):
    global tolerance_a
    tolerance_a = 0.1 + 0.1 * val
    print("a:", tolerance_a)


def change_tolerance_b(val):
    global tolerance_b
    tolerance_b = 0.5 + 0.1 * val
    print("b:", tolerance_b)


def change_sigma(val):
    global sigma
    sigma = val / 100


def change_cast_level(val):
    global cast_factor
    cast_factor = val / 30


def remove_green_cast(out):
    b, g, r = cv2.split(out)
    rb = cast_factor * (r + b)
    new_g = np.where(g > rb, rb, g)

    return cv2.merge((b, new_g, r))


def show_instructions():
    print("1. Select a region by drawing a rectangle from the frame")
    print("2. Option 1: Press `b` to mark the region as definite background")
    print("2. Option 2: Press `f` to mark the region as definite foreground")
    print("3. You go back to step 1 to keep selecting and assigning as many times as you want")
    print("4. If you have assigned the last selected region press `q` to go to the next step")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Chroma Keying with Mishima's Algorithm"
    )

    parser.add_argument(
        'input', type=str,
        help='path to the input image/video'
    )
    parser.add_argument(
        'bg', type=str,
        help='path to the input background image'
    )

    args = parser.parse_args()

    # cap = cv2.VideoCapture("greenscreen-demo.mp4")
    cap = cv2.VideoCapture(args.input)
    new_bg = cv2.imread(args.bg)

    win_name = 'composite'
    win_selection_name = 'selection'

    cv2.namedWindow(win_name)
    cv2.namedWindow(win_selection_name)

    cv2.createTrackbar("Tolerance slider (a)", win_name, 0, 30, change_tolerance_a)
    cv2.createTrackbar("Tolerance slider (b)", win_name, 0, 30, change_tolerance_b)
    cv2.createTrackbar("Softness", win_name, 0, 100, change_sigma)
    cv2.createTrackbar("Color cast", win_name, 0, 30, change_cast_level)

    cv2.setTrackbarPos("Tolerance slider (a)", win_name, 0)
    cv2.setTrackbarPos("Tolerance slider (b)", win_name, 0)
    cv2.setTrackbarPos("Color cast", win_name, 13)
    cv2.setTrackbarPos("Softness", win_name, 88)

    cv2.setMouseCallback(win_selection_name, select_regions)

    FPS = 60
    # So 1/.025 = 40 FPS
    TIMEOUT = 1 / FPS
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    processing_size = tuple([s // 4 for s in size])
    new_bg = cv2.resize(new_bg, processing_size)
    new_bg = new_bg.astype(np.float32) / 255.0
    # bg = cv2.cvtColor(bg, cv2.COLOR_BGR2YCrCb)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    show_instructions()

    ret, frame = cap.read()
    frame = cv2.resize(frame, processing_size)
    frame_copy = frame.copy()
    while True:
        cv2.imshow(win_selection_name, frame_copy)
        key_pressed = cv2.waitKey(20)
        if not REGION_SELECTION_MODE and key_pressed == ord('b'):
            rect = LAST_REGION.pop()
            BG_REGIONS.append(rect)
            REGION_SELECTION_MODE = True

        elif not REGION_SELECTION_MODE and key_pressed == ord('f'):
            rect = LAST_REGION.pop()
            FG_REGIONS.append(rect)
            REGION_SELECTION_MODE = True

        elif len(BG_REGIONS) > 0 and len(FG_REGIONS) > 0 and REGION_SELECTION_MODE and key_pressed == ord('q'):
            break

        elif not REGION_SELECTION_MODE and key_pressed == ord('q'):
            print("[ERROR]: Please assign a region before exiting.")
            show_instructions()

        elif REGION_SELECTION_MODE and key_pressed in [ord('f'), ord('b')]:
            print("[ERROR]: Please select a rectangular regoin before assigning it")
            show_instructions()

        elif (len(BG_REGIONS) < 1 or len(FG_REGIONS) < 1) and key_pressed == ord('q'):
            print("[ERROR]: At least one background and at least one foreground have to be assigned")
            show_instructions()

    bg_regions = [(np.arange(y, y+h), np.arange(x, x+w)) for x, y, w, h in BG_REGIONS]
    fg_regions = [(np.arange(y, y+h), np.arange(x, x+w)) for x, y, w, h in FG_REGIONS]

    bg_coords = [np.meshgrid(rows, cols) for rows, cols in bg_regions]
    fg_coords = [np.meshgrid(rows, cols) for rows, cols in fg_regions]

    bg_indices = np.concatenate(
        [np.ravel_multi_index([x.flatten(), y.flatten()],
                                       frame.shape[:2]) for x, y in bg_coords]
    )

    fg_indices = np.concatenate(
        [np.ravel_multi_index([x.flatten(), y.flatten()],
                                       frame.shape[:2]) for x, y in fg_coords]
    )
    bg_indices = np.unique(bg_indices)
    fg_indices = np.unique(fg_indices)

    frame_linear = frame.reshape((-1, 3))
    bg_samples = frame_linear[bg_indices].astype(np.float32)

    bg_mean = bg_samples.mean(axis=0)

    bg_samples_idx = np.random.choice(bg_indices, size=(1000,))
    fg_samples_idx = np.random.choice(fg_indices, size=(10000,))

    bg_samples = frame_linear[bg_samples_idx].astype(np.float32)
    fg_samples = frame_linear[fg_indices].astype(np.float32) #frame_linear[fg_samples_idx].astype(np.float32)

    # cv2.imshow("bg_mask", bg_mask.astype(np.float32))
    # cv2.imshow("fg_mask", fg_mask.astype(np.float32))

    centered_fg = fg_samples - bg_mean
    centered_bg = bg_samples - bg_mean
    mishima.determine_inner_hexoctahedron(bg_samples)
    mishima.determine_outer_hexoctahedron(fg_samples)
    mishima.compute_all_plane_params()

    cv2.destroyWindow(win_selection_name)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, processing_size)

        mask = mishima.compute_vlahos_mask(frame, tolerance_a, tolerance_b)

        mask = (mask - mask.min())
        mask /= mask.max()

        if DEBUG:
            plt.hist(mask.flatten(), bins=100)
            plt.savefig("hist.png")

        mask = mask.flatten()
        mask_bg = mask[bg_indices]
        mask_fg = mask[fg_indices]

        mask = mask.reshape(frame.shape[:2])
        mask[mask >= mask_fg.min()] = 1.0
        mask[mask <= mask_bg.max()] = 0.0

        if DEBUG:
            cv2.imshow('old_mask', mask)

        bg_mask = mask < 0.2
        fg_mask = mask > 0.8
        unknown_mask = np.logical_and(~bg_mask, ~fg_mask)

        temp = np.zeros_like(mask)
        temp[unknown_mask] = 1.00

        if DEBUG:
            cv2.imshow('unk_mask', temp)

        if unknown_mask.sum() > 0:
            unk = frame[unknown_mask].reshape((-1, 3)).astype(np.float32)

            centered_unk = unk - bg_mean

            unk_alpha = mishima.compute_alpha(centered_unk)
            alpha = mask.copy()
            alpha[unknown_mask] = unk_alpha
        else:
            alpha = mask

        if DEBUG:
            cv2.imshow('new mask', alpha)

        alpha = np.expand_dims(alpha, 2)
        alpha = np.repeat(alpha, 3, axis=2)
        frame = frame.astype(np.float32) / 255.0

        alpha = cv2.GaussianBlur(alpha, (3, 3), sigma)
        out = alpha * frame + (1 - alpha) * new_bg
        out = remove_green_cast(out)

        cv2.imshow(win_name, out)
        if DEBUG:
            cv2.imshow('fr', frame)

        if cv2.waitKey(20) == ord('q'):
            break

cv2.destroyAllWindows()