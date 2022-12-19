import glob
import numpy as np
from PIL import Image

import os
import torch
import matplotlib.pyplot as plt
from torchmetrics import F1Score, Accuracy, Precision, Recall
from scipy.io import loadmat
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import mediapipe as mp
import cv2
import shutil
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


np.random.rand(42)


class data_preprocessing:
    def __init__(self, mat_path, train_test_split=0.8, folder_directory="Data\imdb_crop"):
        self.meta = loadmat(mat_path)
        self.index_to_remove = []
        self.train_test_split = train_test_split
        self.folder_directory = folder_directory

    def __extract_year_from_matlab_datenum(self, matlab_datenum, index):
        """Extract year from matlab datenum

        Args:
            matlab_datenum (float): matlab datenum
            index (int): index of the matlab datenum

        Returns:
            int: year
        """
        try:
            python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        except:
            self.index_to_remove.append(index)

        return python_datetime.year

    def split_df(self, df):
        """Split the data into train and test

        Args:
            df (pd.DataFrame): meta data

        Returns:
            tuple: train and test dataframes
        """
        train = df.sample(frac=self.train_test_split, random_state=42)
        test = df.drop(train.index)

        return train, test

    def __convert_matlab_datenum_array(self, matlab_datenum_array):
        """Convert matlab datenum array to dob date

        Args:
            matlab_datenum_array (np.array): matlab datenum array

        Returns:
            np.array: python datetime array
        """

        dob = []
        for i, date in enumerate(matlab_datenum_array):
            try:
                year = self.__extract_year_from_matlab_datenum(date, i)
            except:
                pass
            dob.append(year)

        return dob

    @staticmethod
    def calculate_age(dob, photo_taken_date):
        """Calculate the age of the person in the image

        Args:
            dob (int): date of birth
            photo_taken_date (int): photo taken date

        Returns:
            int: age
        """
        return photo_taken_date - dob

    def get_label_df(self, face_score_threshold=0.5, second_face_score_threshold=2.5):
        """Get the meta data from the mat file

        Returns:
            pd.DataFrame: meta data
        """

        matlab_dob = self.meta["imdb"][0][0][0].ravel().astype(np.float64)
        date_photo_taken = self.meta["imdb"][0][0][1].ravel().astype(np.float64)
        image_paths = self.meta["imdb"][0][0][2].ravel()
        gender = self.meta["imdb"][0][0][3].ravel()
        celeb_id = self.meta["imdb"][0][0][9].ravel().astype(np.float64)
        face_locations = self.meta["imdb"][0][0][5].ravel()

        # Convert matlab datenum to python datetime
        dob = self.__convert_matlab_datenum_array(matlab_dob)
        age = self.calculate_age(dob, date_photo_taken)
        image_paths = [x.tolist()[0] for x in image_paths]

        face_score = self.meta["imdb"][0][0][6].ravel()
        second_face_score = self.meta["imdb"][0][0][7].ravel()
        self.index_to_remove += list(np.argwhere(second_face_score > second_face_score_threshold).ravel())
        self.index_to_remove += list(np.argwhere(np.isnan(gender)).ravel())  # Remove nan values for gender
        self.index_to_remove += list(np.argwhere(age < 0).ravel())  # Remove negative ages
        self.index_to_remove += list(np.argwhere(age > 100).ravel())  # Remove extreme ages
        self.index_to_remove += list(np.argwhere(face_score < face_score_threshold).ravel())  # Remove face scores below a certain threshold

        self.df = pd.DataFrame({"image_path": image_paths, "age": age, "gender": gender, "face_score": face_score, "face_locations": face_locations, "celeb_id": celeb_id})
        # Create cropped_images and remove non-rgb images
        self.create_cropped_images()

        self.df = self.df.drop(self.index_to_remove)

        return self.df[["image_path", "age", "gender"]]

    def create_cropped_images(self):
        """Create cropped images from the original images using mediapipes face detection and remove non-rgb images"""

        mp_face_detection = mp.solutions.face_detection

        sub_dirs = set(self.df["image_path"].apply(lambda x: x.split("/")[0]))
        for sub_dir in sub_dirs:
            # check if the sub_dir exists, if not create it
            if not os.path.exists(os.path.join(self.folder_directory + "_processed", sub_dir)):
                os.makedirs(os.path.join(self.folder_directory + "_processed", sub_dir))
            # If the directory already exists, delete it and create a new one
            else:
                shutil.rmtree(os.path.join(self.folder_directory + "_processed", sub_dir))
                os.makedirs(os.path.join(self.folder_directory + "_processed", sub_dir))

        IMAGE_FILES = list("Data/imdb_crop/" + self.df["image_path"])
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=1) as face_detection:
            print("Creating cropped images...")
            for idx, file in enumerate(tqdm(IMAGE_FILES, total=len(IMAGE_FILES))):
                image = cv2.imread(file)

                image_rows, image_cols, num_channels = image.shape

                # Remove non-rbg images
                if num_channels != 3:
                    self.index_to_remove.append(idx)
                    continue
                elif (image[:, :, 0] == image[:, :, 1]).all() or (image[:, :, 0] == image[:, :, 2]).all() or (image[:, :, 1] == image[:, :, 2]).all():
                    self.index_to_remove.append(idx)
                    continue
                else:
                    pass

                # Process image with MediaPipe Face Detection.
                results = face_detection.process(image)

                if not results.detections:
                    self.index_to_remove.append(idx)
                    continue
                relative_bounding_box = results.detections[0].location_data.relative_bounding_box

                try:
                    # Convert the bounding box from relative coordinates to pixel coordinates.
                    rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows)
                    rect_end_point = _normalized_to_pixel_coordinates(
                        relative_bounding_box.xmin + relative_bounding_box.width, relative_bounding_box.ymin + relative_bounding_box.height, image_cols, image_rows
                    )

                    # get bounding box coordinates
                    xleft = int(rect_start_point[0])
                    ytop = int(rect_start_point[1])
                    xright = int(rect_end_point[0])
                    ybottom = int(rect_end_point[1])

                    # increase the bounding box by 10%
                    xleft = max(int(xleft - 0.1 * (xright - xleft)), 0)
                    ytop = max(int(ytop - 0.3 * (ybottom - ytop)), 0)
                    xright = min(int(xright + 0.1 * (xright - xleft)), image_cols)
                    ybottom = min(int(ybottom + 0.1 * (ybottom - ytop)), image_rows)

                    crop_img = image[ytop:ybottom, xleft:xright]
                    cv2.imwrite("imdb_crop_processed".join(file.split("imdb_crop")), crop_img)
                except:
                    # bounding box failed, most likely because the the face is at the edge of the image
                    self.index_to_remove.append(idx)
                    continue

    @staticmethod
    def count_labels(labels):
        """Count the number of labels in a list of labels

        Args:
            labels (list): list of labels

        Returns:
            dict: dictionary containing the number of labels
        """
        label_count = {}
        for label in labels:

            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        return label_count

    # -------------------------- plots --------------------------

    def plot_face_score(self):
        """Plot the face score distribution

        Args:
            df (pd.DataFrame): meta data
        """
        face_scores = self.meta["imdb"][0][0][6].ravel()
        sns.displot(face_scores)
        plt.savefig(r"Data/Plots/face_score.png")

    def plot_gender_distribution(self):

        genders = self.df["gender"]
        gender_lookup = {0: "Female", 1: "Male"}
        gender_counts = self.count_labels(genders)
        labels = [gender_lookup[i] for i in list(gender_counts.keys())]

        # Bar plot
        fig, ax = plt.subplots()
        ax.bar(x=[1, 2], height=gender_counts.values(), tick_label=labels)
        plt.savefig(r"Data/Plots/gender_split.png")

    def plot_age_distribution(self):
        """Plot the age distribution

        Args:
            df (pd.DataFrame): meta data
        """
        age = self.meta["imdb"][0][0][1].ravel()
        sns.countplot(x=age)
        plt.savefig(r"Data/Plots/age_distribution.png")


if __name__ == "__main__":
    processing = data_preprocessing("Data\Metadata\imdb.mat")
    df = processing.get_label_df()
    train, test = processing.split_df(df)

    processing.plot_face_score()
    processing.plot_gender_distribution()

    train.to_csv("Data\Metadata\meta_data_train.csv", index=False)
    test.to_csv("Data\Metadata\meta_data_test.csv", index=False)
