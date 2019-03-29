# Emotion Recognition
## Objectives
Implement a VGG-like network capable of predicting emotion and facial expressions:
* Converted and splited `fer2013.csv` dataset into three `HDF5` files including training, validation, and testing sets.
* Constructed a VGG-like network from scratch.
* Train an emotion recognizer and improve the model accuracy.
* Evaluate the emotion recognizer trained previously.
* Built a real-time application in detecting a person's emotion/facial expression.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.1.0
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/scipylib/index.html)

## Approaches
The dataset, called `fer2013`, comes from [Kaggle Emotion and Facial Expression Recognition challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge). The training dataset has 28,709 images, each of which are 48x48 grayscale images. The faces have been automatically aligned such that they are approximately the same size in each image. Given these images, my goal is to categorize the emotion expressed on each face into six distinct classes: angry, fear, happy, sad, surprise, and neutral.

The original dataset also has a seventh classes named disgust, which only has about 113 image samples (the rest have over 1000 image samples per class). After doing some research, I decide to merge both "disgust" and "angry" into a single class ([reference](https://github.com/JostineHo/mememoji)) at latest experiment.

### Build fer2013 dataset
The `emotion_config.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/config/emotion_config.py)) inside `config/` directory, stores several configuration variables, including paths to the input dataset, output `HDF5` files, and batch sizes.

The `build_dataset.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/build_dataset.py)) is responsible for ingesting the `fer2013.csv` dataset file and outputting set a set of HDF5 files, one for each of the training, validation, and testing splits, respectively.

We can use the following command line to build datasets.
```
python build_dataset.py
```

### Construct VGG-like network from scratch
The network architecture is inspired by the family of VGG networks:
1. All the `CONV` layers in the network will be 3x3.
2. The number of filters learned by each `CONV` layer will be doubled as the network become deeper.

Table 1 below shows the network architecture. The activation and batch normalization layers are not shown in the table, which should be after each `CONV` and `FC` layer (`CONV` => Activation => BN, and `FC` => Activation => BN). The both `relu` and `elu` activation functions are used in the project. The `emotionvggnet.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/pipeline/nn/conv/emotionvggnet.py)) inside `pipeline/nn/conv/` directory is responsible for constructing the architecture.

| Layer Type    | Output Size   | Filter Size / Stride  |
| ------------- |:-------------:| ---------------------:|
| Input Image   | 48 x 48 x 1   | 3 x 3, K = 32         |
| CONV          | 48 x 48 x 32  | 3 x 3, K = 32         |
| CONV          | 48 x 48 x 32  | 3 x 3, K = 32         |
| POOL          | 24 x 24 x 32  | 2 x 2                 |
| Dropout       | 24 x 24 x 32  |                       |
| CONV          | 24 x 24 x 64  | 3 x 3, K = 64         |
| CONV          | 24 x 24 x 64  | 3 x 3, K = 64         |
| POOL          | 12 x 12 x 64  | 2 x 2                 |
| Dropout       | 12 x 12 x 64  |                       |
| CONV          | 12 x 12 x 128 | 3 x 3, K = 128        |
| CONV          | 12 x 12 x 128 | 3 x 3, K = 128        |
| POOL          | 6 x 6 x 128   | 2 x 2                 |
| Dropout       | 6 x 6 x 128   |                       |
| FC            | 64            |                       |
| Dropout       | 64            |                       |
| FC            | 64            |                       |
| FC            | 6             |                       |
| softmax       | 6             |                       |

Table 1: The EmotionVGGNet architecture.

### Train and evaluate the VGG-like network
The `train_recognizer.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/train_recognizer.py)) (latest version) is responsible for training the emotion recognizer.

If we want to train the network from the beginning, we can use the following command line to train the network:
```
python train_recognizer.py --checkpoints checkpoints
```

If we want to continue training or retrain the network from some specific epoch, we can use the following command the line to resume the training process (replace `{number_of_the_epoch_to_start}` with a number):
```
python train_recognizer --checkpoints checkpoints --model checkpoints/epoch_{number_of_the_epoch_to_start}.hdf5 --start_epoch {number_of_the_epoch_to_start}
```

The `test_recognizer.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/test_recognizer.py)) is responsible for evaluating the network by using testing set.

The following command will evaluate the network at specific epoch by using testing set, simply replace the `{number_of_the_epoch}` with a number.
```
python test_recognizer.py --model checkpoints/epoch_{number_of_the_epoch}.hdf5
```

### Build a real-time application in detecting the a person's emotion
The `emotion_detector.py` ([check here](https://github.com/meng1994412/Emotion_Recognition/blob/master/emotion_detector.py)) is used to detect a person's emotion either in a live stream camera or a video. `haarcascade_fromtalface_default.xml` is used to help us localize the frontal face of a person.

The following command line can apply the emotion detector on a live stream camera.
```
python emotion_detector.py --cascade haarcascade_fromtalface_default.xml --model checkpoints/model.hdf5
```

Or apply the emotion detector on a video
```
python emotion_detector.py --cascade haarcascade_fromtalface_default.xml --model checkpoints/model.hdf5 --video video.mp4
```

## Results
### Train and evaluate the VGG-like network
#### Experiment 1
I decide to establish a baseline model for the first experiment. Thus in experiment #1, I use `SGD` optimizer with base learning rate of 0.01, a momentum term of 0.9, and Nesterov acceleration applied. The `Xavier/Glorot` initialization method is used to initialize the weights in `CONV` and `FC` layers. For the data augmentation, I only apply horizontal flip. And `relu` activation function is used.

Table 1 illustrates the Learning rate schedule for the experiment #1.

Table 1: Learning rate schedule for experiment #1.

|Epoch   | Learning Rate|
|:------:|:------------:|
|1 - 20  | 1e-2         |
|21 - 40 | 1e-3         |
|41 - 60 | 1e-4         |

Figure 1 shows the accuracy and loss plot of training and validation, which obtains about 63.51% validation accuracy.

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_1.png" width="500">

Figure 1: Accuracy and loss of training and validation for experiment #1, reaching 63.51% validation accuracy.

#### Experiment 2
As we can see from the Figure 1 (epochs 20 - 60), `SGD` leads to stagnation even when dropping the learning rate from 0.001 to 0.0001. Thus I change the optimizer to `Adam` with base learning of 0.001, leaving everything else unchanged.

Figure 2 demonstrate the accuracy and loss plot of training and validation. I stop the training at 40 epochs because there is a sign showing that overfitting might occur and lower the learning rate to 0.0001 and resume the training for another 15 epochs. Now it's clear that severe overfitting occurs, though the validation accuracy is about 66.85%.

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_2.png" width="500">

Figure 2: Accuracy and loss of training and validation for experiment #2, which suffers from overfitting, though the validation accuracy is 66.85%.

#### Experiment 3
In the experiment #3, I add more data augmentation parameters including random rotation range of 10 degrees along with zoom range of 0.1 to reduce the overfitting. And I use the learning rate schedule shown in Table 2. Figure 3 demonstrate the accuracy and loss of training and validation for experiment #3, which reaches validation accuracy about 67.55%.

Table 2: Learning rate schedule for experiment #3.

|Epoch   | Learning Rate|
|:------:|:------------:|
|1 - 40  | 1e-3         |
|41 - 60 | 1e-4         |
|61 - 75 | 1e-5         |

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_3.png" width="500">

Figure 3: Accuracy and loss of training and validation for experiment #3, which reaches 67.55% validation accuracy.

#### Experiment 4
In the experiment #4, I make two major changes, including:
* change `relu` activation function to `elu` activation function to further boost up the accuracy.
* change `Xavier/Glorot` initialization method to `He/MSRA` initialization method, since `He/MSRA` initialization method tends to work better for the VGG family.

Figure 4 shows the accuracy and loss of training and validation for experiment #4, which obtains 68.45% validation accuracy.

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_4.png" width="500">

Figure 4: Accuracy and loss of training and validation for experiment #4, which obtains 68.45% validation accuracy.

#### Experiment 5
After experiment #4, I wonder if I could make the network deeper to further boost the accuracy. Thus, I add another block of 2 * (CONV => Activation => BN) => POOL => Dropout in the network, as Table 3 shown below.

| Layer Type    | Output Size   | Filter Size / Stride  |
| ------------- |:-------------:| ---------------------:|
| Input Image   | 48 x 48 x 1   | 3 x 3, K = 32         |
| CONV          | 48 x 48 x 32  | 3 x 3, K = 32         |
| CONV          | 48 x 48 x 32  | 3 x 3, K = 32         |
| POOL          | 24 x 24 x 32  | 2 x 2                 |
| Dropout       | 24 x 24 x 32  |                       |
| CONV          | 24 x 24 x 64  | 3 x 3, K = 64         |
| CONV          | 24 x 24 x 64  | 3 x 3, K = 64         |
| POOL          | 12 x 12 x 64  | 2 x 2                 |
| Dropout       | 12 x 12 x 64  |                       |
| CONV          | 12 x 12 x 128 | 3 x 3, K = 128        |
| CONV          | 12 x 12 x 128 | 3 x 3, K = 128        |
| POOL          | 6 x 6 x 128   | 2 x 2                 |
| Dropout       | 6 x 6 x 128   |                       |
| CONV          | 6 x 6 x 256   | 3 x 3, K = 256        |
| CONV          | 6 x 6 x 256   | 3 x 3, K = 256        |
| POOL          | 3 x 3 x 128   | 2 x 2                 |
| Dropout       | 3 x 3 x 128   |                       |
| FC            | 64            |                       |
| Dropout       | 64            |                       |
| FC            | 64            |                       |
| FC            | 6             |                       |
| softmax       | 6             |                       |

Table 3: The deeper EmotionVGGNet architecture.

Since the network becomes deeper, overfitting might occur again, so I add `L2` regularization in the network of 0.0001. Other than this, everything else stays the same.

The Figure 5 shows the accuracy and loss of training and validation for experiment #5, which obtains 69.71% validation accuracy. However, overfitting still occurs (epochs 50 - 70).

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_5.png" width="500">

Figure 5: Accuracy and loss of training and validation for experiment #6, which obatins 69.71% validation accuracy but still suffers from overfitting.

#### Experiment 6
In order to reduce the overfitting occuring in experiment #5, I decide to increase the `L2` regularization term to 0.0005 and keep everything the same.

The Figure 6 shows the accuracy and loss of training and validation for experiment #6, which obtains 69.83% validation accuracy. Now, the overfitting is largely reduced (epochs 50 - 70).

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/vggnet_emotion_6.png" width="500">

Figure 6: Accuracy and loss of training and validation for experiment #6, which obatins 69.83% validation accuracy but still suffers from overfitting.

**By using the testing set to evaluate the model, the accuracy is 68.72%, as Figure 7 shown, which can claim #4 position in the Leaderboard.**

<img src="https://github.com/meng1994412/Emotion_Recognition/blob/master/output/evaluation_6.png" width="200">

Figure 7: Evaluation of the network by using testing set.
