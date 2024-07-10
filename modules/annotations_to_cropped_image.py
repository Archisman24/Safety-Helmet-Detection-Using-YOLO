import cv2
import os

def annot2crop(filepath, classes, classes2crop=None, padding=30):
    # Read the image
    image = cv2.imread(filepath)
    lh, lw, _ = image.shape

    # Get the corresponding txt file location
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(filename)[0]

    txt_file_path = os.path.join(directory, '../labels', filename_without_extension + '.txt')

    # Create a directory for saving the cropped images
    cropped_images_dir = os.path.join(directory, '../Cropped_images')
    os.makedirs(cropped_images_dir, exist_ok=True)

    # Iterate over the contents of the txt file
    with open(txt_file_path, 'r') as file:
        counter = 0
        for line in file:
            ls = line.strip().split()
            if classes2crop is None:
                class_dir = os.path.join(cropped_images_dir, 'All_classes')
            elif ls[0] in classes2crop:
                class_dir = os.path.join(cropped_images_dir, classes[ls[0]])
            else:
                class_dir = os.path.join(cropped_images_dir, 'Others')

            os.makedirs(class_dir, exist_ok=True)

            x, y, w, h = map(float, ls[1:5])  # Convert coordinates to float
            x, y, w, h = int(x * lw), int(y * lh), int(w * lw), int(h * lh)  # Adjust coordinates

            # Expand the cropping region based on the padding value
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding

            # Check if cropping region is within image bounds
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > lw:
                w = lw - x
            if y + h > lh:
                h = lh - y

            # Check if cropping region is valid
            if w <= 0 or h <= 0:
                continue

            boxedImage = image[y:y+h, x:x+w]
            cropped_file_path = os.path.join(class_dir, f'{filename_without_extension}_{counter}.jpg')
            cv2.imwrite(cropped_file_path, boxedImage)
            counter += 1

    # Return to the original directory
    os.chdir(directory)


