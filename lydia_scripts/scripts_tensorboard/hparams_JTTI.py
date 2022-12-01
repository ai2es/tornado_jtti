# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Write sample summary data for the hparams plugin.
See also `hparams_minimal_demo.py` in this directory for a demo that
runs much faster, using synthetic data instead of actually training
MNIST models.
"""

#GRAB GPU0
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])

import os.path
import random
import shutil
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import xarray as xr 
from tensorboard.plugins.hparams import api as hp
import sys
import matplotlib.pyplot as plt
import io
sys.path.append("/home/lydiaks2/tornado_project")




if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


flags.DEFINE_integer(
    "num_session_groups",
    100,
    "The approximate number of session groups to create.",
)

flags.DEFINE_string(
    "logdir",
    "/tmp/hparams_demo",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    200,
    "Number of epochs per trial.",
)


# Define data size
INPUT_SHAPE = (32,32,12)
OUTPUT_CLASSES = 2


# Define Hyperparameters
#data params
HP_DATA_PATCHES_TYPE = hp.HParam("data_patches_type", hp.Discrete(['tor','nontor_tor']))


#convolution params
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.Discrete([1, 2, 3]))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(['ReLU', 'LeakyReLU', 'PReLU', 'ELU',]))
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([4,8,16,32]))
HP_L1_REGULARIZATION = hp.HParam('l1 regularization', hp.Discrete([1e-1,1e-2,1e-3,1e-4,1e-5]))
HP_L2_REGULARIZATION =  hp.HParam('l2 regularization', hp.Discrete([1e-1,1e-2,1e-3,1e-4,1e-5]))


#unet param
HP_UNET_TYPE = hp.HParam('type_of_unet', hp.Discrete(['unet_2d','unet_plus_2d','unet_3plus_2d']))
HP_UNET_DEPTH = hp.HParam('depth_of_unet', hp.Discrete([1,2,3,4]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "adagrad","sgd","rmsprop"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(["categorical_crossentropy", "fractions_skill_score", "binary_focal_crossentropy"]))
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([True,False]))
# HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([32,64,128,256,512]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2,1e-3,1e-4,1e-5,1e-6]))

# Combine all hparams
HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_UNET_DEPTH,
    HP_OPTIMIZER,
    HP_LOSS,
    HP_BATCHNORM,
    HP_LEARNING_RATE,
    HP_UNET_TYPE,
    HP_DATA_PATCHES_TYPE,
    HP_L1_REGULARIZATION,
    HP_L2_REGULARIZATION,
]

# Define our training metrics
METRICS = [
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
    hp.Metric(
        "epoch_categorical_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "epoch_categorical_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="validation",
        display_name="max csi (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="train",
        display_name="max csi (train)",
    ),
]

# Define our losses for tensorboard
def build_loss_dict():
    from custom_losses import make_fractions_skill_score
    loss_dict = {}
    loss_dict['binary_focal_crossentropy'] = tf.keras.losses.binary_focal_crossentropy
    loss_dict['categorical_crossentropy'] = tf.keras.losses.categorical_crossentropy
    loss_dict['fractions_skill_score'] = make_fractions_skill_score(3, 2, c=1.0, cutoff=0.5, want_hard_discretization=False)
    return loss_dict

# Define our optimizers for tensorboard
def build_opt_dict(learning_rate):
    opt_dict = {}
    opt_dict['adam'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_dict['adagrad'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    opt_dict['sgd'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    opt_dict['rmsprop'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return opt_dict

def model_fn(hparams, seed):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    
    from keras_unet_collection import models

    # Set our random seed
    rng = random.Random(seed)

    # Define our convolutional kernels
    kernel_list = []
    for i in np.arange(1,hparams[HP_UNET_DEPTH]+1,1):
        kernel_list.append(hparams[HP_CONV_KERNELS]*i)

    # Choose the right output activation for our loss function
    if hparams[HP_LOSS] == 'binary_focal_crossentropy':
        output_activation = 'Sigmoid'
        n_labels = 1
    else:
        output_activation = 'Softmax'
        n_labels = OUTPUT_CLASSES

    #Choose the type of unet
    if hparams[HP_UNET_TYPE] == 'unet_2d':
        model = models.unet_2d(INPUT_SHAPE, kernel_list, n_labels=n_labels,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                        stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                        l1=hparams[HP_L1_REGULARIZATION], l2=hparams[HP_L2_REGULARIZATION],
                        activation=hparams[HP_CONV_ACTIVATION], output_activation=output_activation, weights=None,
                        batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unet')
    elif hparams[HP_UNET_TYPE] == 'unet_plus_2d':
        model = models.unet_plus_2d(INPUT_SHAPE, kernel_list, n_labels=n_labels, kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                        stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                        l1=hparams[HP_L1_REGULARIZATION], l2=hparams[HP_L2_REGULARIZATION],
                        activation=hparams[HP_CONV_ACTIVATION], output_activation=output_activation, weights=None,
                        batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unetplus')
    elif hparams[HP_UNET_TYPE] == 'unet_3plus_2d':
        model = models.unet_3plus_2d(INPUT_SHAPE, n_labels=n_labels, filter_num_down=kernel_list, kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                        stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                        l1=hparams[HP_L1_REGULARIZATION], l2=hparams[HP_L2_REGULARIZATION],
                        activation=hparams[HP_CONV_ACTIVATION], output_activation=output_activation, weights=None,
                        batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unet3plus')

    #get metric 
    from custom_metrics import MaxCriticalSuccessIndex

    #compile losses: 
    loss_dict = build_loss_dict()
    opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])

    # Choose the right metrics for our loss function
    if hparams[HP_LOSS] == 'binary_focal_crossentropy':
        metrics=["categorical_accuracy",MaxCriticalSuccessIndex()]
    else:
        metrics=["binary_accuracy",MaxCriticalSuccessIndex()]

    # Build and return the model
    model.compile(
        loss=loss_dict[hparams[HP_LOSS]],
        optimizer=opt_dict[hparams[HP_OPTIMIZER]],
        metrics=metrics,
    )
    return model

def prepare_data(hparams):
    """ Load data """    
    if hparams[HP_LOSS] == 'binary_focal_crossentropy':
        #This loss function needs integer labels
        
        # Define the dataset size
        x_tensor_shape = (None, 32, 32, 12)
        y_tensor_shape = (None, 32, 32, 1)
        elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.int64))

        x_tensor_shape_val = (32, 32, 12)
        y_tensor_shape_val = (32, 32, 1)
        elem_spec_val = (tf.TensorSpec(shape=x_tensor_shape_val, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape_val, dtype=tf.int64))

        # Pick out the correct dataset paths
        if hparams[HP_DATA_PATCHES_TYPE] == 'tor':
            path = "int_tor"
        elif hparams[HP_DATA_PATCHES_TYPE] == 'nontor_tor':
            path = "int_nontor_tor"
    else:
        #These loss functions need onehot labels
        
        # Define the dataset size
        x_tensor_shape = (None, 32, 32, 12)
        y_tensor_shape = (None, 32, 32, 2)
        elem_spec = (tf.TensorSpec(shape=x_tensor_shape, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape, dtype=tf.float32))

        x_tensor_shape_val = (32, 32, 12)
        y_tensor_shape_val = (32, 32, 2)
        elem_spec_val = (tf.TensorSpec(shape=x_tensor_shape_val, dtype=tf.float64), tf.TensorSpec(shape=y_tensor_shape_val, dtype=tf.float32))

        # Pick out the correct dataset paths
        if hparams[HP_DATA_PATCHES_TYPE] == 'tor':
            path = "onehot_tor"
        elif hparams[HP_DATA_PATCHES_TYPE] == 'nontor_tor':
            path = "onehot_nontor_tor"
        
    #Read tensorflow datasets
    ds_train = tf.data.experimental.load("/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/training_" + path + '/training_ZH_only.tf',
                                        elem_spec)
    ds_val = tf.data.experimental.load('/ourdisk/hpc/ai2es/tornado/learning_patches/tensorflow/3D_light/validation_' + path + '/validation1_ZH_only.tf',
                                        elem_spec_val)

    return (ds_train, ds_val)

def run(data, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    # Load in our model
    model = model_fn(hparams=hparams, seed=session_id)
    
    # Define the output path for our logs
    logdir = os.path.join(base_logdir, session_id)

    # Extract our datasets
    ds_train,ds_val = data
    ds_val = ds_val.batch(512)

    # Define our callbacks
    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )
    hparams_callback = hp.KerasCallback(logdir, hparams)
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    #add images to board 
    print(model.weights[0].shape)
    print(model.summary())
    result = model.fit(ds_train,
        epochs=flags.FLAGS.num_epochs,
        shuffle=False,
        validation_data=ds_val,
        callbacks=[callback, hparams_callback,callback_es],verbose=0)

    #Build the path to save the trained model
    split_dir = logdir.split('jtti')
    split_dir2 = split_dir[1].split('logs')
    right = split_dir2[0][:-1] + split_dir2[1]
    left = '/scratch/lydiaks2/models/jtti/'
    model.save(left + right + "model.h5")


def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    # Set the random seed
    rng = random.Random(0)

    # Open tensorboard logs
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        
        # Open our datasets
        data = prepare_data(hparams)
        
        # Run tensorboard
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))
            run(
                data=data,
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
            )


def main(unused_argv):
    np.random.seed(0)
    logdir = flags.FLAGS.logdir
    print('removing old logs')
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    app.run(main)