# import packages
from config import emotion_config as config
from pipeline.io import HDF5DatasetWriter
import numpy as np

# open the input file for reading (skip the header)
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()

# initialize the list of data and labels for training, validation and testing sets
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# loop over the rows in the input file
for row in f:
    # extract the label, image, and usage from the row
    (label, image, usage) = row.strip().split(",")
    label = int(label) # change label str type to int type

    # if we are ignoring the "disgust" class there will be 6 classes
    if config.NUM_CLASSES == 6:
        # merge together the "anger" and "disgust" classes
        if label == 1:
            label = 0

        # if label has a value greater than zero, subtract one from it to
        # make all labels sequential
        if label > 0:
            label -= 1

    # reshape the flattened pixel list into 48x48 grayscale image
    image = np.array(image.split(" "), dtype = "uint8")
    image = image.reshape((48, 48))

    # check if we are examing a training image
    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    # check if we are examing a validation image
    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    # otherwise, it must be a testing image
    else:
        testImages.append(image)
        testLabels.append(label)

# construct a list pairing the training, validation, and testing images
# along with their corresponding labels and output HDF5 files
datasets = [
    (trainImages, trainLabels, config.TRAIN_HDF5),
    (valImages, valLabels, config.VAL_HDF5),
    (testImages, testLabels, config.TEST_HDF5)
]

# loop over dataset tuples
for (images, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    # loop over the image and add them to the datatset
    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    # close the HDF5 writer
    writer.close()

# close the input file
f.close()
