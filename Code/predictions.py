import tensorflow as tf
from librosa import feature
import librosa
import numpy as np
import math
import matplotlib.pyplot as plt

input_path='hungwe 50 sec sample.wav'

model = tf.keras.models.load_model('my rnn model.keras')



def predict(model, X):
    labels_to_name={0:'noise',1:'normal',2:'laugh',3:'pain',4:'hunger'}

    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    predicted_index=int(predicted_index)
    print(" Predicted label: {}".format( labels_to_name[predicted_index]))

def split_audio(input_file,segment_duration=3):
    # Load the audio file
    y, sr = librosa.load(input_file)

    # Calculate the number of samples in each segment
    segment_samples = int(segment_duration * sr)

    # Calculate the total number of segments
    num_segments = int(math.ceil(len(y) / segment_samples))

    # Split the audio into segments
    for i in range(num_segments):
        start_sample = i * segment_samples
        end_sample = min((i + 1) * segment_samples, len(y))

        # Extract the segment
        segment = y[start_sample:end_sample]
        mfcc = feature.mfcc(y=segment, sr=22050, n_mfcc=23, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        mfcc = mfcc[..., np.newaxis]
        predict(model,mfcc)


split_audio(input_path)



