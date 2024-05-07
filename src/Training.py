import numpy as np
import pandas as pd
import os
from skimage.feature import hog
import joblib
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from common import clean
from PIL import Image


class Training:
    def __init__(self, model_path: str, pixels_per_cell: int = 2, cells_per_block: int = 1, orientation: int = 9,
                 train_iterations: int = 100):
        self.model_path = model_path
        self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        self.cells_per_block = (cells_per_block, cells_per_block)
        self.orientation = orientation
        self.train_iterations = train_iterations

    def create_dataset(self, dirpath: str) -> pd.DataFrame:
        # Read all labels and pictures in labels directory
        labels_list = [filename for filename in os.listdir(dirpath)]
        pics_paths = [os.listdir(os.path.join(dirpath, filename)) for filename in labels_list]

        # Separate features(HOG of pics) and labels(Text in pics for prediction)
        features, labels = [], []
        for i, j in enumerate(zip(pics_paths, labels_list)):
            imgs, label = j
            for img in imgs:
                # Read picture from disk and clean it
                img = Image.open(os.path.join(dirpath, label, img))
                img = clean(img)

                # Convert PIL image to cv2.Image
                img = np.asarray(img)[:, :, ::-1]
                img_res = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
                img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
                hog_img = hog(img_gray, orientations=self.orientation, pixels_per_cell=self.pixels_per_cell,
                              cells_per_block=self.cells_per_block)
                features.append(hog_img)
                labels.append(label)

        # create dataframe from features and labels
        df = pd.DataFrame(np.array(features))
        df['target'] = labels
        return df

    def split_train_test(self, dataset: pd.DataFrame, balance_dataset: bool = True):
        x = np.array(dataset.iloc[:, :-1])
        y = np.array(dataset['target'])
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.20,
                                                            random_state=42)

        if balance_dataset:
            sm = SMOTE(random_state=0)
            x_train, y_train = sm.fit_resample(x_train, y_train)
            return x_train, y_train, x_test, y_test

        return x_train, y_train, x_test, y_test

    def train(self, dataset_dir: str):
        # create dataset from provided directory path
        dataset = self.create_dataset(dataset_dir)

        # split dataset in train and test phase
        x_train, y_train, x_test, y_test = self.split_train_test(dataset=dataset, balance_dataset=True)

        # train our model
        logistic_regression = LogisticRegression(max_iter=self.train_iterations)
        model = logistic_regression.fit(x_train, y_train)

        # save trained model to disk
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)


if __name__ == '__main__':
    model = Training(model_path="../data/models/voters.eci.gov.in.pkl", pixels_per_cell=2, cells_per_block=1,
                     orientation=9, train_iterations=10000)

    model.train(dataset_dir="../data/dataset")