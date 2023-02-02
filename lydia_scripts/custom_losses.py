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

# ###LOSSES FROM HERE: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py

# beta = 0.25
# alpha = 0.25
# gamma = 2
# epsilon = 1e-5
# smooth = 1


# import tensorflow as tf 
# import tensorflow.keras.backend as K
# from keras.losses import binary_crossentropy

# beta = 0.25
# alpha = 0.25
# gamma = 2
# epsilon = 1e-5
# smooth = 1


# class Semantic_loss_functions(object):
#     def __init__(self,build_dict=True):
#         if build_dict:
#           loss_dict = {}
#           loss_dict['dice_loss'] = self.dice_loss
#           loss_dict['weighted_cross_entropyloss'] = self.weighted_cross_entropyloss
#           loss_dict['focal_loss'] = self.focal_loss
#           loss_dict['bce_dice_loss'] = self.bce_dice_loss
#           loss_dict['tversky_loss'] = self.tversky_loss
#           self.losses = loss_dict
            
#     def dice_coef(self, y_true, y_pred):
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#         intersection = K.sum(y_true_f * y_pred_f)
#         return (2. * intersection + K.epsilon()) / (
#                     K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

#     def sensitivity(self, y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         return true_positives / (possible_positives + K.epsilon())

#     def specificity(self, y_true, y_pred):
#         true_negatives = K.sum(
#             K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#         possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
#         return true_negatives / (possible_negatives + K.epsilon())

#     def convert_to_logits(self, y_pred):
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
#                                   1 - tf.keras.backend.epsilon())
#         return tf.math.log(y_pred / (1 - y_pred))

#     def weighted_cross_entropyloss(self, y_true, y_pred):
#         y_pred = self.convert_to_logits(y_pred)
#         pos_weight = beta / (1 - beta)
#         loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
#                                                         labels=y_true,
#                                                         pos_weight=pos_weight)
#         return tf.reduce_mean(loss)

#     def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
#         weight_a = alpha * (1 - y_pred) ** gamma * targets
#         weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

#         return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
#             -logits)) * (weight_a + weight_b) + logits * weight_b

#     def focal_loss(self, y_true, y_pred):
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
#                                   1 - tf.keras.backend.epsilon())
#         logits = tf.math.log(y_pred / (1 - y_pred))

#         loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
#                                       alpha=alpha, gamma=gamma, y_pred=y_pred)

#         return tf.reduce_mean(loss)

#     def depth_softmax(self, matrix):
#         sigmoid = lambda x: 1 / (1 + K.exp(-x))
#         sigmoided_matrix = sigmoid(matrix)
#         softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
#         return softmax_matrix

#     def generalized_dice_coefficient(self, y_true, y_pred):
#         smooth = 1.
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#         intersection = K.sum(y_true_f * y_pred_f)
#         score = (2. * intersection + smooth) / (
#                     K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#         return score

#     def dice_loss(self, y_true, y_pred):
#         loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
#         return loss

#     def bce_dice_loss(self, y_true, y_pred):
#         loss = binary_crossentropy(y_true, y_pred) + \
#                self.dice_loss(y_true, y_pred)
#         return loss / 2.0

#     def confusion(self, y_true, y_pred):
#         smooth = 1
#         y_pred_pos = K.clip(y_pred, 0, 1)
#         y_pred_neg = 1 - y_pred_pos
#         y_pos = K.clip(y_true, 0, 1)
#         y_neg = 1 - y_pos
#         tp = K.sum(y_pos * y_pred_pos)
#         fp = K.sum(y_neg * y_pred_pos)
#         fn = K.sum(y_pos * y_pred_neg)
#         prec = (tp + smooth) / (tp + fp + smooth)
#         recall = (tp + smooth) / (tp + fn + smooth)
#         return prec, recall

#     def true_positive(self, y_true, y_pred):
#         smooth = 1
#         y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#         y_pos = K.round(K.clip(y_true, 0, 1))
#         tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
#         return tp

#     def true_negative(self, y_true, y_pred):
#         smooth = 1
#         y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#         y_pred_neg = 1 - y_pred_pos
#         y_pos = K.round(K.clip(y_true, 0, 1))
#         y_neg = 1 - y_pos
#         tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
#         return tn

#     def tversky_index(self, y_true, y_pred):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = K.sum(y_true_pos * y_pred_pos)
#         false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#         false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
#         alpha = 0.7
#         return (true_pos + smooth) / (true_pos + alpha * false_neg + (
#                     1 - alpha) * false_pos + smooth)

#     def tversky_loss(self, y_true, y_pred):
#         return 1 - self.tversky_index(y_true, y_pred)

#     def focal_tversky(self, y_true, y_pred):
#         pt_1 = self.tversky_index(y_true, y_pred)
#         gamma = 0.75
#         return K.pow((1 - pt_1), gamma)

#     def log_cosh_dice_loss(self, y_true, y_pred):
#         x = self.dice_loss(y_true, y_pred)
#         return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

# class WeightedMSE(tf.keras.losses.Loss):
#     """ 
#     Calculate a weighted MSE. This loss gives you control to weight the 
#     pixels that are > 0 differently than the pixels that are 0 in y_true. This
#     class is subclassed from tf.keras.lossess.Loss to hopefully enable 
#     'stateful'-ness ?

#     weights[0] is the weight for non-zero pixels
#     weights[1] is the weight for zero pixels. 

#     """
#     def __init__(self, weights=[1.0,1.0],name="custom_mse",
#                  **kwargs):
#         super(WeightedMSE,self).__init__(name=name, **kwargs)

#         #store weights
#         self.w1 = weights[0]
#         self.w2 = weights[1]

#     def call(self, y_true, y_pred):

#         #build weight_matrix 
#         ones_array = tf.ones_like(y_true)
#         weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
#         weights_for_zero = tf.math.multiply(ones_array,self.w2)
#         weight_matrix = tf.where(tf.greater(y_true,0),weights_for_nonzero,
#                            weights_for_zero)
#         loss = tf.math.reduce_mean(tf.math.multiply(weight_matrix,
#                                                     tf.math.square(tf.math.subtract(y_pred,y_true))))
#         return loss

# class RegressLogLoss(tf.keras.losses.Loss):
#     """Baseline regression log-loss that includes uncertainty via a 2-output regression setup.
#     """
#     def __init__(self,):
#         super().__init__()

#     def __str__(self):
#         return ("RegressBaseline()")

#     def call(self, y_true, y_pred):
        
#         y_true = tf.cast(y_true, tf.float64)
#         y_pred = tf.cast(y_pred, tf.float64)
        
#         mu = y_pred[...,0]
#         sigma = tf.math.exp(y_pred[...,1])
        
#         norm_dist = tfp.distributions.Normal(mu, sigma)

#         # compute the log as -log(p)
#         loss = -norm_dist.log_prob(y_true[...,0])
        
#         return tf.reduce_mean(loss)