# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # Splitting the data
    sentences = bbc['text'].tolist()
    labels = bbc['category'].tolist()

    # Split the data into training and validation sets
    training_size = int(len(sentences) * training_portion)

    # YOUR CODE HERE
    # Using "shuffle=False"
    training_sentences = sentences[0:training_size]
    validation_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    validation_labels = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences) # YOUR CODE HERE
    
    # You can also use Tokenizer to encode your label.
    # Tokenize and pad the training and validation sentences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                      truncating=trunc_type)

    # Tokenize the labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)

    training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define a Callback class that stops training once accuracy and validation accuracy are > 91%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):
                print("\nReached 91% accuracy and validation accuracy so stopping training!")
                self.model.stop_training = True

    callbacks = myCallback()

    # Train the model
    history = model.fit(training_padded, training_label_seq, epochs=30,
                        validation_data=(validation_padded, validation_label_seq), verbose=2, callbacks=[callbacks])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")