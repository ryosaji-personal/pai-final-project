import cv2
import shutil
from pathlib import Path


# ==========================================
# 設定
# ==========================================

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw"

LABELS = {
    ord("0"): "none",
    ord("1"): "left",
    ord("2"): "center",
    ord("3"): "right",
}

WINDOW_NAME = "Image Labeling"

# 画像の表示倍率
DISPLAY_SCALE = 1.0


# ==========================================
# 画像一覧
# ==========================================

def get_image_files():

    return sorted(
        list(RAW_DIR.glob("*.png"))
    )


# ==========================================
# 画像表示
# ==========================================

def show_image(image_path, remaining, counts):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot read: {image_path}")
        return False

    height, width = image.shape[:2]

    display = image.copy()

    # 左・中央・右の境界線
    x1 = width // 3
    x2 = width * 2 // 3

    cv2.line(
        display,
        (x1, 0),
        (x1, height),
        (255, 255, 0),
        1
    )

    cv2.line(
        display,
        (x2, 0),
        (x2, height),
        (255, 255, 0),
        1
    )

    # 操作説明の表示
    cv2.putText(
        display,
        "0:None  1:Left  2:Center  3:Right",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display,
        "U:Undo  S:Skip  Q:Quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # 処理状況
    status = (
        f"Remaining: {remaining}  "
        f"N:{counts['none']} "
        f"L:{counts['left']} "
        f"C:{counts['center']} "
        f"R:{counts['right']}"
    )

    cv2.putText(
        display,
        status,
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # 表示倍率を適用
    if DISPLAY_SCALE != 1.0:

        display = cv2.resize(
            display,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE
        )

    cv2.imshow(WINDOW_NAME, display)

    return True


# ==========================================
# ラベル枚数の集計
# ==========================================

def count_labels():

    counts = {}

    for label in LABELS.values():

        directory = DATASET_DIR / label

        counts[label] = len(
            list(directory.glob("*.png"))
        )

    return counts


# ==========================================
# メイン処理
# ==========================================

def main():

    # ディレクトリ作成
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for label in LABELS.values():

        (DATASET_DIR / label).mkdir(
            parents=True,
            exist_ok=True
        )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # 直前の移動履歴
    history = []

    # スキップした画像
    skipped = set()

    print("Image labeling started.")
    print("0:None 1:Left 2:Center 3:Right")
    print("u:Undo s:Skip q:Quit")

    try:

        while True:

            # 未分類画像を毎回取得
            image_files = [
                p for p in get_image_files()
                if p not in skipped
            ]

            if not image_files:

                remaining = get_image_files()

                if remaining:
                    print(
                        f"\n{len(remaining)} images skipped."
                    )
                else:
                    print("\nAll images labeled!")

                break

            image_path = image_files[0]

            counts = count_labels()

            remaining = len(get_image_files())

            if not show_image(
                image_path,
                remaining,
                counts
            ):
                skipped.add(image_path)
                continue

            print(
                f"\nImage: {image_path.name}"
                f" | Remaining: {remaining}"
            )

            # キー入力待ち
            key = cv2.waitKey(0) & 0xFF

            # ----------------------------------
            # ラベル指定
            # ----------------------------------

            if key in LABELS:

                label = LABELS[key]

                destination = (
                    DATASET_DIR
                    / label
                    / image_path.name
                )

                # 同名ファイルが存在したら上書きしない
                if destination.exists():

                    print(
                        f"ERROR: Destination exists: "
                        f"{destination}"
                    )

                    skipped.add(image_path)
                    continue

                shutil.move(
                    str(image_path),
                    str(destination)
                )

                history.append(
                    (image_path, destination)
                )

                print(
                    f"Moved: {image_path.name} "
                    f"-> {label}"
                )

            # ----------------------------------
            # Undo
            # ----------------------------------

            elif key == ord("u"):

                if not history:

                    print("Nothing to undo.")
                    continue

                source, destination = history.pop()

                if source.exists():
                    print(
                        f"Cannot undo: {source} exists."
                    )
                    continue

                shutil.move(
                    str(destination),
                    str(source)
                )

                print(
                    f"Undo: {destination.name}"
                )

                # 戻した画像を次に表示
                skipped.clear()

            # ----------------------------------
            # Skip
            # ----------------------------------

            elif key == ord("s"):

                skipped.add(image_path)

                print(
                    f"Skipped: {image_path.name}"
                )

            # ----------------------------------
            # Quit
            # ----------------------------------

            elif key == ord("q"):

                print("Labeling interrupted.")
                break

            else:

                print("Invalid key.")

    finally:

        cv2.destroyAllWindows()

    # 最終集計
    counts = count_labels()

    remaining = len(get_image_files())

    print("\n===== Labeling Summary =====")

    for label, count in counts.items():
        print(f"{label:8s}: {count}")

    print(f"Remaining: {remaining}")


if __name__ == "__main__":
    main()
