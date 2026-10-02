import cv2
import os

OUTPUT_DIR = "tests/camera_previews"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Checking camera indices 0-5...\n")
    found_any = False
    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if ret and frame is not None:
            found_any = True
            path = os.path.join(OUTPUT_DIR, f"camera_{index}.jpg")
            cv2.imwrite(path, frame)
            print(f"Index {index}: WORKS ({width}x{height}) -- saved preview to {path}")
        else:
            print(f"Index {index}: opened but could not read a frame")
    if not found_any:
        print("\nNo working cameras found. Make sure Camo Studio is running and your phone is connected before running this script.")
    else:
        print(f"\nOpen the images in {OUTPUT_DIR}/ and find the one showing your phone view -- that number is the index to use with main.py")

if __name__ == "__main__":
    main()
