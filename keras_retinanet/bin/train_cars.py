import os
import sys

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

import tensorflow as tf
from .. import losses
from .. import models
from ..callbacks import RedirectModel, keras
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.car_generator import CarsGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_car_generators(root_path, preprocess_image, batch_size):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': batch_size,
        'config': None,
        'image_min_side': 800,
        'image_max_side': 1333,
        'preprocess_image': preprocess_image,
    }

    transform_generator = random_transform_generator(flip_x_chance=0.5)

    train_generator = CarsGenerator(root_path,
                                    transform_generator=transform_generator,
                                    **common_args
                                    )

    validation_generator = None
    return train_generator, validation_generator


def main(root_path, snapshot_file=None, backbone_name='resnet50', lr=0.01, epochs=50, freeze_backbone=True, batch_size=16):
    # create object that stores backbone information
    backbone = models.backbone(backbone_name)

    # make sure keras is the minimum required version
    check_keras_version()

    # create the generators
    train_generator, validation_generator = create_car_generators(root_path, backbone.preprocess_image, batch_size)

    steps_per_epoch = train_generator.size() / batch_size

    # create the model
    if snapshot_file is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(snapshot_file, backbone_name=backbone_name)
        training_model = model
        anchor_params = None
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=0,
            freeze_backbone=freeze_backbone,
            lr=lr
        )

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        batch_size
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                               weights=weights, skip_mismatch=True)
    training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_callbacks(model,
                     training_model,
                     prediction_model,
                     validation_generator,
                     batch_size,
                     snapshot_path='./',
                     backbone='resnet50',
                     tensorboard_dir='logs/'):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    # if args.evaluation and validation_generator:
    #     if args.dataset_type == 'coco':
    #         from ..callbacks.coco import CocoEval
    #
    #         # use prediction model for evaluation
    #         evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
    #     else:
    #         evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback,
    #                               weighted_average=args.weighted_average)
    #     evaluation = RedirectModel(evaluation, prediction_model)
    #     callbacks.append(evaluation)

    # save the model
    if snapshot_path:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=backbone,
                                                                    dataset_type='cars')
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks


main('../')