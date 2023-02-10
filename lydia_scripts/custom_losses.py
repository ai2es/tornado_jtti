import tensorflow as tf 
import tensorflow.keras.backend as K
from keras.losses import binary_crossentropy
# import tensorflow_probability as tfp


##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

# Fraction Skill Score (FSS) Loss Function - code taken from: https://github.com/CIRA-ML/custom_loss_functions
# Fraction Skill Score original paper: N.M. Roberts and H.W. Lean, "Scale-Selective Verification of Rainfall
#     Accumulation from High-Resolution Forecasts of Convective Events", Monthly Weather Review, 2008.
def make_fractions_skill_score(mask_size, num_dimensions, c=1.0, cutoff=0.5, want_hard_discretization=False):
    """
    Make fractions skill score loss function. Visit https://github.com/CIRA-ML/custom_loss_functions for documentation.
    Parameters
    ----------
    mask_size: int or tuple
        - Size of the mask/pool in the AveragePooling layers.
    num_dimensions: int
        - Number of dimensions in the mask/pool in the AveragePooling layers.
    c: int or float
        - C parameter in the sigmoid function. This will only be used if 'want_hard_discretization' is False.
    cutoff: float
        - If 'want_hard_discretization' is True, y_true and y_pred will be discretized to only have binary values (0/1)
    want_hard_discretization: bool
        - If True, y_true and y_pred will be discretized to only have binary values (0/1).
        - If False, y_true and y_pred will be discretized using a sigmoid function.
    Returns
    -------
    fractions_skill_score: float
        - Fractions skill score.
    """

    pool_kwargs = {'pool_size': mask_size}
    if num_dimensions == 2:
        pool1 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling2D(**pool_kwargs)
    elif num_dimensions == 3:
        pool1 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
        pool2 = tf.keras.layers.AveragePooling3D(**pool_kwargs)
    else:
        raise ValueError("Number of dimensions can only be 2 or 3")

    @tf.function()
    def fractions_skill_score(y_true, y_pred):
        """ Fractions skill score loss function """
        if want_hard_discretization:
            y_true_binary = tf.where(y_true > cutoff, 1.0, 0.0)
            y_pred_binary = tf.where(y_pred > cutoff, 1.0, 0.0)
        else:
            y_true_binary = tf.math.sigmoid(c * (y_true - cutoff))
            y_pred_binary = tf.math.sigmoid(c * (y_pred - cutoff))

        y_true_density = pool1(y_true_binary)
        n_density_pixels = tf.cast((tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]), tf.float32)

        y_pred_density = pool2(y_pred_binary)

        MSE_n = tf.keras.metrics.mean_squared_error(y_true_density, y_pred_density)

        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)

        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels

        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if want_hard_discretization:
            if MSE_n_ref == 0:
                return MSE_n
            else:
                return MSE_n / MSE_n_ref
        else:
            return MSE_n / (MSE_n_ref + my_epsilon)

    return fractions_skill_score

def csi(use_as_loss_function, use_soft_discretization,
            hard_discretization_threshold=None):
            
        def loss(target_tensor, prediction_tensor):
            if hard_discretization_threshold is not None:
                prediction_tensor = tf.where(
                    prediction_tensor >= hard_discretization_threshold, 1., 0.
                )
            elif use_soft_discretization:
                prediction_tensor = K.sigmoid(prediction_tensor)
        
            num_true_positives = K.sum(target_tensor * prediction_tensor)
            num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
            num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
            denominator = (
                num_true_positives + num_false_positives + num_false_negatives +
                K.epsilon()
            )

            csi_value = num_true_positives / denominator
            
            if use_as_loss_function:
                return 1. - csi_value
            else:
                return csi_value
        
        return loss

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Adapted tf.keras.backend.binary_crossentropy to have weights that work for a UNET
    
    Last I checked, weighting the binary cross entropy wouldnt work with the UNETs, it
    use to complain about the shape being not right. 
    
    To get around this, the class here has a way to weight the 1s and 0s differently.
    This is important for alot of meteorological instances where the event we are looking 
    to predict is often a 'rare' event. We mean where the pixels == 1 are like 1% of the 
    total number of pixels. 
    
    the expected shape of y_true is [n_sample,nx,ny,1]
    
    """
    def __init__(self,weights=[1.0,1.0],from_logits=False):
        super().__init__()
        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.from_logits=from_logits

    def __str__(self):
        return ("WeightedBinaryCrossEntropy()")
    
    def binary_crossentropy(self,target, output):
        """Binary crossentropy between an output tensor and a target tensor.
        Args:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
        Returns:
          A tensor.
        """
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)

        # Use logits whenever they are available. `softmax` and `sigmoid`
        # activations cache logits on the `output` Tensor.
        if hasattr(output, '_keras_logits'):
            output = output._keras_logits  # pylint: disable=protected-access
            if self.from_logits:
              warnings.warn(
                  '"`binary_crossentropy` received `from_logits=True`, but the `output`'
                  ' argument was produced by a sigmoid or softmax activation and thus '
                  'does not represent logits. Was this intended?"',
                  stacklevel=2)
            self.from_logits = True

        if self.from_logits:
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        if (not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable)) and
          output.op.type == 'Sigmoid') and not hasattr(output, '_keras_history'):
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        epsilon_ = _constant_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

        # Compute cross entropy from probabilities.
        bce = target * tf.math.log(output + tf.keras.backend.epsilon())
        bce += (1 - target) * tf.math.log(1 - output + tf.keras.backend.epsilon()) 
        return -bce

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        loss = self.binary_crossentropy(y_true, y_pred)
    
        #build weight_matrix 
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,0.0),weights_for_nonzero,
                           weights_for_zero)
        self.w = weight_matrix
        #weight non-zero pixels more
        loss = tf.math.multiply(weight_matrix,loss)
    
        return tf.reduce_mean(loss)