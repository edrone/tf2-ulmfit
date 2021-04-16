import tensorflow as tf
from .almost_ulmfit_tf2 import almost_ulmfit_model


def ulmfit_quick_and_dirty_sequence_tagger(*, num_classes=3, pretrained_weights='wagi'):
    print("Building model from Python code (not tf.saved_model)...")
    au = almost_ulmfit_model()
    print("Restoring weights from file....")
    au.load_weights(pretrained_weights)
    print("Removing LM head")
    au.pop() # remove final dropout
    au.pop() # remove LM head
    print(f"Adding sequence tagging head with n_classes={num_classes}")
    au.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax')))
    return au
