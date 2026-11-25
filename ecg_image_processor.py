import cv2
import numpy as np


def load_ecg_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read ECG image. "
            "Please upload a valid JPG, JPEG or PNG image."
        )

    return image


def create_waveform_mask(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 60, 40])
    upper_red_1 = np.array([15, 255, 255])

    lower_red_2 = np.array([165, 60, 40])
    upper_red_2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    red_mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = cv2.inRange(
        gray,
        20,
        110
    )

    mask = cv2.bitwise_or(
        red_mask,
        dark_mask
    )

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def extract_waveform(crop):
    mask = create_waveform_mask(crop)

    height, width = mask.shape

    signal = np.full(
        width,
        np.nan,
        dtype=np.float32
    )

    for x in range(width):

        ys = np.where(
            mask[:, x] > 0
        )[0]

        if len(ys) == 0:
            continue

        center = height / 2

        distances = np.abs(
            ys - center
        )

        selected_y = ys[
            np.argmin(distances)
        ]

        signal[x] = selected_y

    valid = ~np.isnan(signal)

    if np.sum(valid) < width * 0.15:
        return None

    x_all = np.arange(width)

    x_valid = x_all[valid]

    y_valid = signal[valid]

    signal = np.interp(
        x_all,
        x_valid,
        y_valid
    )

    signal = -signal

    signal = signal - np.mean(signal)

    maximum = np.max(
        np.abs(signal)
    )

    if maximum > 0:
        signal = signal / maximum

    signal = cv2.resize(
        signal.reshape(-1, 1),
        (1000, 1),
        interpolation=cv2.INTER_LINEAR
    )[0]

    signal = signal.astype(
        np.float32
    )

    return signal


def process_ecg_image(image_path):
    image = load_ecg_image(
        image_path
    )

    height, width = image.shape[:2]

    print(
        "Original ECG image:",
        width,
        "x",
        height
    )

    rows = 3
    cols = 4

    margin_x = int(
        width * 0.03
    )

    margin_y = int(
        height * 0.08
    )

    usable_width = (
        width
        - 2 * margin_x
    )

    usable_height = (
        height
        - 2 * margin_y
    )

    cell_width = (
        usable_width // cols
    )

    cell_height = (
        usable_height // rows
    )

    leads = []

    for row in range(rows):

        for col in range(cols):

            x1 = (
                margin_x
                + col * cell_width
            )

            x2 = (
                margin_x
                + (col + 1) * cell_width
            )

            y1 = (
                margin_y
                + row * cell_height
            )

            y2 = (
                margin_y
                + (row + 1) * cell_height
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            crop_h = crop.shape[0]

            top_crop = int(
                crop_h * 0.15
            )

            bottom_crop = int(
                crop_h * 0.92
            )

            crop = crop[
                top_crop:bottom_crop,
                :
            ]

            signal = extract_waveform(
                crop
            )

            if signal is None:

                raise ValueError(
                    "Could not extract waveform "
                    f"from ECG lead position "
                    f"{row + 1}-{col + 1}. "
                    "Please upload a clear 12-lead ECG image."
                )

            leads.append(
                signal
            )

    ecg_signal = np.stack(
        leads,
        axis=1
    )

    if ecg_signal.shape != (
        1000,
        12
    ):

        raise ValueError(
            "Invalid processed ECG shape: "
            f"{ecg_signal.shape}"
        )

    print(
        "Processed ECG shape:",
        ecg_signal.shape
    )

    return ecg_signal