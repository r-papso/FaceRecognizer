# FaceRecognizer

FaceRecognizer is a Python demo application used for face recognition. I has been created as a part of my [bachelor thesis](https://opac.crzp.sk/?fn=detailBiblioForm&sid=81B8C7E3C61A011F1630311BFB16&seo=CRZP-detail-kniha) and consists of simple GUI, SQLite database used as storage for identities and trained convolutional neural network (CNN) with same architecture as [Light-CNN network](https://arxiv.org/abs/1511.02683).

## Prerequisites

Libraries needed to be installed before using FaceRecognizer app:

  * [Keras](https://keras.io/) (2.3.1)
  * [Tensorflow](https://www.tensorflow.org/) (2.1.0)
  * [Dlib](http://dlib.net/python/index.html) (19.19.0)
  * [Numpy](https://numpy.org/) (1.18.1)
  * [Pillow](https://python-pillow.org/) (7.0.0)
  * [OpenCV](https://opencv.org/) (4.2.0)
  * [Skimage](https://scikit-image.org/) (0.16.2)
  
Numbers in parentheses are versions of libraries used in development of the application. You can use different versions, but if you do so, application running is not guaranteed.

## Running the app

You can run the application from command line by executing [main.py](main.py) file:

```bash
python main.py
```
