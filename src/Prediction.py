from PIL import Image
import os
from skimage.feature import hog
import joblib
import numpy as np
import pandas as pd
from common import clean


class Prediction:
    def __init__(self, model_path: str, pixels_per_cell: int = 2, cells_per_block: int = 1, orientation: int = 9):
        self.clf = joblib.load(model_path)
        self.pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        self.cells_per_block = (cells_per_block, cells_per_block)
        self.orientation = orientation

    def predict(self, picture: Image) -> str:
        # clean captcha image before prediction and convert to grayscale
        picture = clean(picture).convert("L")

        # co-ordinates for character cropping
        coordinates = [(20, 0, 50, picture.height),
                       (50, 0, 75, picture.height - 20),
                       (75, 0, 110, picture.height),
                       (100, 0, 130, picture.height - 30),
                       (123, 0, 153, picture.height),
                       (145, 0, picture.width, picture.height),
                       ]
        # Get all characters in this file
        prediction = ""
        for index in range(0, 6):
            letter = picture.crop(coordinates[index])
            # Resize the image
            roi = letter.resize((64, 128), Image.Resampling.LANCZOS)
            # Calculate the HOG features
            # use the same parameters used for training
            roi_hog_fd = hog(roi, orientations=self.orientation, pixels_per_cell=self.pixels_per_cell,
                             cells_per_block=self.cells_per_block)
            nbr = self.clf.predict(np.array([roi_hog_fd]))
            prediction += str(nbr[0])

        return prediction


if __name__ == '__main__':
    # initialize prediction class once
    predictor = Prediction(
        model_path="../data/models/voters.eci.gov.in.pkl",
        pixels_per_cell=2,
        cells_per_block=1,
        orientation=9)

    # Store values for visualization in html format
    filenames, previews, predictions = [], [], []

    # Read all files to predict result
    basepath = "../data/captchas"
    for index, filename in enumerate(os.listdir(basepath)):
        # Exit if we have more than 500 images for this test(optional)
        if index >= 100:
            break

        filepath = os.path.join(basepath, filename)

        # Read image from filepath and perform prediction
        picture = Image.open(filepath)
        predicted = predictor.predict(picture=picture)

        # Store Prediction in a file to preview its results
        filenames.append(filename)
        previews.append(picture)
        predictions.append(predicted)
        print(f"Completed: {index+1}/100 files")

    # Create a dataframe with few columns
    df = pd.DataFrame(columns=["filename", "preview", "prediction"])
    df["filename"] = filenames
    df["preview"] = previews
    df["prediction"] = predictions

    # Rest everything is just for generating html to see
    from io import BytesIO
    import base64


    def get_thumbnail(path):
        i = Image.open(path)
        i.thumbnail((150, 150), Image.LANCZOS)
        return i


    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()


    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


    with open("../data/previews.html", 'w') as file:
        html = df.to_html(formatters={'preview': image_formatter}, escape=False)
        file.write(html)
    print("Output saved to previews.html file")
