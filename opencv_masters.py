import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from dataset import download_dataset
from features import FeatureExtractor

UPLOAD_FOLDER = "./uploads/"
ZIP_FOLDER = "./data/zips"
DATA_FOLDER = "./data/images"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "bmp"])

app = Flask(__name__)

# store the UPLOAD_FOLDER constant in the app.config dictionary so
# we can access it easily
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def train_classifier(X, Y):
    clf = SVC(kernel='linear', C=0.025)
    clf.fit(X, Y) # X is our feature matrix, Y is our label vector
    return clf


def visualise(X):
    model = TSNE(n_components=2, random_state=0)
    Y = model.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    plt.show()


@app.route('/', methods=["POST"])
def upload_file():
    """
        the @app.route(url, methods) decorator defines the url location and the
        methods we can use to consume the route.

        this method only allows POST requests as its sole purpose is to receive
        images.
        
        if everything goes ok, this route will save the provided file to disk
        and return a unique id for the file which we can use later.
    """
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            # status 500 means there is a server side error.
            # google HTTP codes for more info
            return jsonify({"status": 500, "message": "no file in request"})
        file = request.files["file"]

        # if the user didn't select a file, the browser
        # will send an empty file with and empty filename.
        if file.filename == '':
            return jsonify({"status": 500, "message": "empty file in request"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # status 200 means everything is ok
            return jsonify({"status": 200, "message": "ok", "filename": filename})


clf = None
# this is like the python equivalent of the main() function.
# app.run() starts our server
if __name__ == '__main__':

    # make sure we have the dataset (this might take a while!)
    download_dataset(ZIP_FOLDER, DATA_FOLDER)

    # convert our dataset into feature vectors and labels
    image_list = [os.path.join(DATA_FOLDER, fpath) for fpath in os.listdir(DATA_FOLDER)]
    fe = FeatureExtractor(image_list)

    # let's define a 80/20 train/test split (approx)
    train_end_point = int(len(image_list) * 0.8)

    # get the train features and train labels
    train_features = fe.feature_matrix[0:train_end_point, :]
    train_labels = fe.label_array[0:train_end_point]

    # get the test features and test labels
    test_features = fe.feature_matrix[train_end_point:, :]
    test_labels = fe.label_array[train_end_point:]

    visualise(test_features)

    # now let's train a classifier!
    clf = train_classifier(train_features, train_labels)

    # now let's see how accurate it is
    predicted_classes = clf.predict(test_features)

    print("pred", predicted_classes)
    print("test", test_labels)


    num_correct = np.sum(predicted_classes == test_labels)
    print(num_correct, (num_correct * 100) / len(test_labels), "%")

    # run the server
    app.run()
