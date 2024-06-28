# Rename images by create time
import os
import glob


if __name__ == '__main__':
    # cropped images folder
    raw_images_root = r"D:\data\Downloads\gte\wc1"

    # sort with create time
    raw_images_list = sorted(
        glob.glob(os.path.join(raw_images_root, "*.png")),
        key=lambda file_path: os.path.getctime(file_path))

    start_id = 24
    is_male = True
    for image_path in raw_images_list:
        if is_male:
            os.rename(image_path, os.path.join(raw_images_root, f"{start_id}_0.png"))
            is_male = False  # next image should be female
        else:
            os.rename(image_path, os.path.join(raw_images_root, f"{start_id}_1.png"))
            is_male = True  # next image should be male
            start_id += 1