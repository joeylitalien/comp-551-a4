# -*- coding: utf-8 -*-
""" image.py: LIME for images """

# Set paths for TF-Silm
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"
sys.path.append("../models/slim")
# Tensorflow
import tensorflow as tf
slim = tf.contrib.slim
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet
# Third-party
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
import time
# LIME
from lime import lime_image

session = tf.Session()
image_size = inception.inception_v3.default_image_size

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f).read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]

names = imagenet.create_readable_names_for_imagenet_labels()
processed_images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
probabilities = tf.nn.softmax(logits)

checkpoints_dir = "../models/slim/pretrained"
init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, "inception_v3.ckpt"),
    slim.get_model_variables("InceptionV3"))
init_fn(session)

def predict_fn(images):
    return session.run(probabilities, feed_dict={processed_images: images})

images = transform_img_fn(["../data/cat-dog.jpg"])
# I"m dividing by 2 and adding 0.5 because of how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
preds = predict_fn(images)
for x in preds.argsort()[0][-5:]:
    print x, names[x], preds[0,x]

image = images[0]
tmp = time.time()
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)
print time.time() - tmp
temp, mask = explanation.get_image_and_mask(208, positive_only=False, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.savefig("../data/output.jpg")
