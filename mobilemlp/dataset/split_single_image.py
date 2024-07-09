import os
import glob
import shutil
import cv2
import json


def main():
    for jpg_path in glob.glob(os.path.join(root, "*.jpg")):
        if os.path.isfile(f"{os.path.splitext(jpg_path)[0]}.json"):
            shutil.copy(jpg_path, split_folder)
            shutil.copy(f"{os.path.splitext(jpg_path)[0]}.json", split_folder)
        else:
            shutil.copy(jpg_path, single_folder)


def png2jpg():
    jpg_root = r"E:\media\single"
    for jpg_path in glob.glob(os.path.join(jpg_root, "*.jpg")):
        png_path = os.path.splitext(jpg_path)[0]+".png"
        cv2.imwrite(png_path, cv2.imread(jpg_path))


def read_json():
    json_root = r"E:\media\split"
    output_root = r"E:\media\output"
    prev_id = 210
    for json_path in glob.glob(os.path.join(json_root, "*.json")):
        with open(json_path, encoding="utf-8") as fp:
            data = json.load(fp)
            for index, shape in enumerate(data["shapes"]):
                if index % 2 == 0:
                    prev_id += 1
                point0, point1 = shape["points"]
                point0 = list(map(int, point0))
                point1 = list(map(int, point1))
                x0, x1, y0, y1 = min(point0[0], point1[0]), max(point0[0], point1[0]), min(point0[1], point1[1]), max(point0[1], point1[1])
                full_image = cv2.imread(os.path.splitext(json_path)[0]+".jpg")
                if shape["label"] == "male":
                    target_path = os.path.join(output_root, f"{prev_id}_0.png")
                else:
                    target_path = os.path.join(output_root, f"{prev_id}_1.png")
                print(json_path)
                cv2.imwrite(target_path, full_image[y0:y1, x0:x1])


if __name__ == '__main__':
    root = r"E:\media\toilet"
    split_folder = r"E:\media\split"
    single_folder = r"E:\media\single"

    # main()

    read_json()
