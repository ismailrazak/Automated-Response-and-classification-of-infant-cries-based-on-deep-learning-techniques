import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE=22050
path='TESTING'
output_folder = 'New'


def split_audio(input_file, output_folder, segment_duration=3,count=0):
    # Load the audio file
    y, sr = librosa.load(input_file)

    # Calculate the number of samples in each segment
    segment_samples = int(segment_duration * sr)

    # Calculate the total number of segments
    num_segments = int(np.ceil(len(y) / segment_samples))

    # Split the audio into segments
    for i in range(num_segments):

        start_sample = i * segment_samples
        end_sample = min((i + 1) * segment_samples, len(y))

        # Extract the segment
        segment = y[start_sample:end_sample]

        # Save the segment to a new file
        output_file = f"{output_folder}/segment_{count}_{i + 1}.wav"

        sf.write(output_file, segment, int(sr))



for  (dirpath,dirname,filename) in os.walk(path):
    for count,f in enumerate(filename):
        audio=os.path.join(dirpath,f)
        split_audio(audio,output_folder,3,count=count)
