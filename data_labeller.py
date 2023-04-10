import cv2
import os
import csv

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
class_map = {k: v for k, v in zip(range(len(classes)), classes)}

path = os.path.join(os.getcwd(), "..", "ADEIJ_datasets", "seg_pred", "seg_pred")


def label(data_path, csv_file="seg_pred_labels.csv"):
    # Open file to append to and make csv writer
    # If the file does not exist, a new one is created automatically
    try:
        with open(csv_file, "r") as f:
            csv_data = csv.reader(f)
            labelled_data = {row[0]: True for row in csv_data}

    except FileNotFoundError:
        labelled_data = {}

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        for data in os.listdir(data_path):
            # Check the existence of the entry in the map so we don't check the whole list every time
            does_exist = False
            try:
                does_exist = labelled_data[data]
            except KeyError:
                labelled_data[data] = True

            if does_exist:
                continue

            # Read image and show it
            current_img = cv2.imread(os.path.join(data_path, data))
            cv2.imshow("image", cv2.resize(current_img, (450, 450)))

            # Sniff user input
            key = int(chr(cv2.waitKey(0)))

            print([data, class_map[key], key])

            # If invalid key is pressed, exit the script
            if key not in range(len(classes)):
                print("Invalid key pressed, exiting script...")
                exit(1)

            # If escape is pressed exit the script
            if cv2.waitKey(0) == 27:
                exit(0)

            # Write the image name and the class information to the csv file
            writer.writerow([data, class_map[key], key])


if __name__ == "__main__":
    label(path)
