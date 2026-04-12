import cv2
import numpy as np


def find_local_peaks(dist_img, min_dist=6, threshold_ratio=0.28):
    """
    在 distance transform 圖上找局部峰值
    min_dist: 峰值之間最小距離
    threshold_ratio: 峰值門檻，相對於 dist_img 最大值
    """
    if dist_img.max() <= 0:
        return []

    thresh = dist_img.max() * threshold_ratio

    # 找局部最大值
    kernel = np.ones((min_dist, min_dist), np.uint8)
    dilated = cv2.dilate(dist_img, kernel)
    local_max = (dist_img == dilated) & (dist_img > thresh)

    ys, xs = np.where(local_max)
    peaks = list(zip(xs, ys, dist_img[ys, xs]))

    # 依峰值強度排序
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)

    selected = []
    for x, y, score in peaks:
        keep = True
        for sx, sy, _ in selected:
            if np.hypot(x - sx, y - sy) < min_dist:
                keep = False
                break
        if keep:
            selected.append((x, y, score))

    return selected


def detect_blue_points(image_path, expected_points=8, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 你的藍點偏亮藍，這組通常可用；必要時可微調
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 去小雜訊
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 稍微侵蝕，讓細線變細，有助於把近點拆開
    mask_eroded = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

    # 找連通區
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_eroded, connectivity=8)

    all_points = []

    for i in range(1, num_labels):  # 0 是背景
        x, y, w, h, area = stats[i]

        # 太小的雜訊略過
        if area < 5:
            continue

        component_mask = np.uint8(labels == i) * 255

        # 只對這個 component 做 distance transform
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)

        # 找局部峰值
        peaks = find_local_peaks(dist, min_dist=6, threshold_ratio=0.28)

        if len(peaks) == 0:
            # 保底：如果沒找到，就用 centroid
            cx, cy = centroids[i]
            all_points.append((float(cx), float(cy), area, 1))
        else:
            # 若同一個 blob 裡找到多個 peak，就代表可能有多個點
            for px, py, score in peaks:
                all_points.append((float(px), float(py), area, score))

    # 全域去重，避免相鄰 component 產生重複點
    all_points = sorted(all_points, key=lambda p: p[3], reverse=True)

    final_points = []
    global_min_distance = 10

    for x, y, area, score in all_points:
        keep = True
        for fx, fy in final_points:
            if np.hypot(x - fx, y - fy) < global_min_distance:
                keep = False
                break
        if keep:
            final_points.append((x, y))

    # 如果超過 expected_points，就保留最合理的 expected_points 個
    # 這裡先依 y 排序只是為了穩定輸出，不代表醫學順序
    if len(final_points) > expected_points:
        final_points = final_points[:expected_points]

    # 存 debug 圖
    debug_img = img.copy()
    for idx, (x, y) in enumerate(final_points, 1):
        cv2.circle(debug_img, (int(round(x)), int(round(y))), 8, (0, 0, 255), 2)
        cv2.putText(debug_img, str(idx), (int(round(x)) + 6, int(round(y)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(f"{debug_prefix}_mask.png", mask)
    cv2.imwrite(f"{debug_prefix}_mask_eroded.png", mask_eroded)
    cv2.imwrite(f"{debug_prefix}_result.png", debug_img)

    return final_points, mask, mask_eroded, debug_img


if __name__ == "__main__":
    image_path = "images/013L'.jpg"   # 改成你的檔名
    points, mask, mask_eroded, debug_img = detect_blue_points(
        image_path,
        expected_points=8,
        debug_prefix="output"
    )

    print("Detected points:")
    for i, (x, y) in enumerate(points, 1):
        print(f"{i}: ({x:.1f}, {y:.1f})")