import cv2
import numpy as np


def make_circle_template(radius=7):
    size = radius * 2 + 5
    template = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    cv2.circle(template, (center, center), radius, 255, -1)
    return template


def nms_points(points, scores, min_dist=10):
    order = np.argsort(scores)[::-1]
    selected = []

    for idx in order:
        x, y = points[idx]
        keep = True
        for sx, sy in selected:
            if np.hypot(x - sx, y - sy) < min_dist:
                keep = False
                break
        if keep:
            selected.append((x, y))
    return selected


def detect_blue_points_template(image_path, expected_points=8, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 藍色遮罩
    lower_blue = np.array([85, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 去雜訊
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 用多種半徑做模板匹配
    radii = [5, 6, 7, 8, 9]
    all_points = []
    all_scores = []

    for r in radii:
        template = make_circle_template(radius=r)
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)

        # 門檻可調
        ys, xs = np.where(result > 0.4)

        for x, y in zip(xs, ys):
            cx = x + template.shape[1] // 2
            cy = y + template.shape[0] // 2
            score = result[y, x]
            all_points.append((cx, cy))
            all_scores.append(score)

    if len(all_points) == 0:
        return [], mask, img

    # NMS 去重
    selected = nms_points(all_points, all_scores, min_dist=6)

    # 保留分數最高的 expected_points 個
    scored_selected = []
    for sx, sy in selected:
        # 找最近的原始候選點分數
        best_score = -1
        for (px, py), sc in zip(all_points, all_scores):
            if np.hypot(px - sx, py - sy) < 3:
                best_score = max(best_score, sc)
        scored_selected.append((sx, sy, best_score))

    scored_selected.sort(key=lambda x: x[2], reverse=True)
    scored_selected = scored_selected[:expected_points]

    final_points = [(x, y) for x, y, _ in scored_selected]

    # debug 圖
    debug_img = img.copy()
    for i, (x, y) in enumerate(final_points, 1):
        cv2.circle(debug_img, (int(x), int(y)), 10, (0, 0, 255), 2)
        cv2.putText(debug_img, str(i), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(f"{debug_prefix}_mask.png", mask)
    cv2.imwrite(f"{debug_prefix}_result.png", debug_img)

    return final_points, mask, debug_img


if __name__ == "__main__":
    image_path = "images/013L'.jpg"   # 改成你的圖
    points, mask, debug_img = detect_blue_points_template(
        image_path,
        expected_points=8,
        debug_prefix="output"
    )

    print("Detected points:")
    for i, (x, y) in enumerate(points, 1):
        print(f"{i}: ({x}, {y})")