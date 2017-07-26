import cv2
import numpy as np
import os

class FeatureExtractor:

    """
        accepts a list of image filenames. 
        this class will read the images from disk and
        produce a feature matrix.
    """

    def __init__(self, image_list):
        assert len(image_list) > 0
        self.image_list = image_list
        self.orb = cv2.ORB_create(250) # x keypoints (at most) to detect per image
        self.build_feature_matrix()

    def build_feature_matrix(self):
        """
            we're going to loop through all images in self.image_list.
            we'll extract some ORB features from each image, storing each
            ORB feature-vector in a normal python array. we'll then
            convert this normal python array into a numpy array.
        """

        print("[info] building feature matrix. got", len(self.image_list), "images")

        # create a normal python array.
        feature_matrix = []
        label_array = []
        for i, fpath in enumerate(self.image_list):
            print("[info] processing image", i, "/", len(self.image_list), fpath)
            # check if the file exists, skip it otherwise
            if os.path.isfile(fpath):
                # the filenames of the swedish leaf dataset follow a
                # specific pattern. e.g. l1nr001.tif means class 1, image 1.
                # we'll use this to figure out which class this image is.

                filename = fpath.split("/")[-1] # split fpath by "/" and get the last element
                class_number = filename.split("n")[0].replace("l", "")

                img = cv2.imread(fpath)
                #img = cv2.resize(img, (1000, 1000))

                # let's do some feature extraction!

                feature_vector = np.zeros((500, 32), dtype='float32') # 500 points x 32 floats (ORB features are 32 floats)

                # orb (http://docs.opencv.org/3.1.0/d1/d89/tutorial_py_orb.html)
                keypoints = self.orb.detect(img, None)
                keypoints, orb_descriptors = self.orb.compute(img, keypoints)

                # sometimes we might get fewer points than the expected 500
                # so we'll check for it, and copy over the features we do have
                if orb_descriptors is not None:
                    if orb_descriptors.shape[0] != 250:
                        num_points = orb_descriptors.shape[0]
                        # orb_descriptors.shape[0] will always be 500 or less (assuming that's what we set the max to be)
                        feature_vector[0:num_points, :] = orb_descriptors

                    # flatten the feature vector and add it to the matrix
                    # flatten turns an [x, y] matrix into a [1, x*y] vector
                    feature_matrix.append(feature_vector.flatten())
                    label_array.append(class_number)
            else:
                print("[error] could not find", fpath)

        self.feature_matrix = np.array(feature_matrix)
        self.label_array = np.array(label_array)


