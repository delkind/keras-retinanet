# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
import math
import os
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet import models


def process_dataset(root):
    train_set = set(open(root + '/ImageSets/test.txt').read().splitlines())
    images = [
        (root + '/Images/' + img, open(root + '/Annotations/' + os.path.splitext(img)[0] + '.txt').read().splitlines())
        for img in os.listdir(root + '/Images') if os.path.splitext(img)[0] in train_set]
    return images


def create_eval_set(remove_zero_boxes):
    images = process_dataset('../datasets/PUCPR+_devkit/data')
    images += process_dataset('../datasets/CARPK_devkit/data')
    images = {k: [[int(n) for n in s.split()] for s in v] for k, v in images}
    images = [(k, len([b for b in v if not remove_zero_boxes or (b[2] - b[0] > 0 and b[3] - b[1] > 0)])) for k, v in images.items()]

    return images


def load_and_process_image(img_path):
    image = read_image_bgr(img_path)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    return image


def predict_image(tup):
    img_path, count = tup

    image = read_image_bgr(img_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale
    preds = [t for t in list(zip(boxes[0], scores[0], labels[0])) if t[1] > 0.5]

    if len(preds) != count:
        title = "{}: Predicted: {}, ground truth: {}".format(img_path, len(preds), count)
        print(title)
    else:
        print("Correct prediction")

    return abs(len(preds) - count)


def visualize_predictions(draw, preds, title):
    # visualize detections
    for box, score, label in preds:
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format('car', score)
        draw_caption(draw, b, caption)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=15)
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


eval_set = create_eval_set(True)
eval_set1 = create_eval_set(False)

diffs = [t[1][1] - t[0][1] for t in zip(eval_set, eval_set1)]

model_path = os.path.join('resnet50_inference.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

errors = list(map(predict_image, eval_set))

print(np.mean(errors), math.sqrt(np.mean([e**2 for e in errors])))

