# COMP 551: Applied Machine Learning
This repository serves for the final assignment of Joelle Pineau's ML course at McGill University, Winter 2017.

## "Why Should I Trust You?": Explaining the Predictions of Any Classifier
Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin

[arXiv link](https://arxiv.org/abs/1602.04938)

## Dependencies
### Modules
* [tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup) (1.0.0-rc2)
* [lime](https://github.com/marcotcr/lime) (latest build)
* [numpy](https://www.scipy.org/scipylib/download.html) (1.12.1)
* [scikit-image](http://scikit-image.org/download.html) (0.13.0)

### TensorFlow Models
Within the project, run:
```sh
$ git clone https://github.com/tensorflow/models
```
To download weights for pretrained ConvNets (e.g. Inception v3), run:
```sh
$ cd models/slim 
$ mkdir pretrained && cd $_
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
$ tar -xvf inception_v3_2016_08_28.tar.gz
$ rm inception_v3_2016_08_28.tar.gz
```
