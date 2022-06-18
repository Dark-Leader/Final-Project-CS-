import cv2
import sys
import os
from algorithm.preprocessing import get_binary_image


def main():
    '''
    process dataset images and transform it to fit model input size and expected image characteristics.
    @return: None.
    '''
    input_folder, output = sys.argv[1], sys.argv[2]
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    new_size = (224, 224)
    count = 0
    for folder in os.listdir(input_folder):
        new_folder_path = f"{parent_dir}/{output}/{folder}"
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
        for i, filename in enumerate(os.listdir(f"{input_folder}/{folder}")):
            path = f"{parent_dir}/{input_folder}/{folder}/{filename}"
            count += 1

            img = cv2.imread(path) # load image.
            try:
                img = get_binary_image(img) # convert image to binary.
                scaled_image = cv2.resize(img, new_size) # scale image to new size.
                new_file_path = f"{parent_dir}/{output}/{folder}/{i+1}.png" # save output image at output folder.
                cv2.imwrite(new_file_path, scaled_image)
            except Exception:
                print(f"Error: file path = {path}") # error processing image.
                pass


if __name__ == '__main__':
    main()
