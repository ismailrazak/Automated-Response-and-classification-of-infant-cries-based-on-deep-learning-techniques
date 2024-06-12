import os
import shutil
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from librosa import feature
import librosa
import numpy as np
import math
import RPi.GPIO as GPIO
from time import sleep,time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import getpass
from picamera2 import Picamera2
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
GPIO.setwarnings(False)

# Right Motor
in1 = 17
in2 = 27
en_a = 12
# Left Motor
in3 = 5
in4 = 6
en_b = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en_a,GPIO.OUT)

GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en_b,GPIO.OUT)

q=GPIO.PWM(en_a,100)
p=GPIO.PWM(en_b,100)
p.start(45)
q.start(45)

GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)



# Load the pre-trained model
model = tf.keras.models.load_model('my rnn model.keras')

# Define the labels
labels_to_name={0:'noise',1:'normal',2:'laugh',3:'pain',4:'hunger'}
# Function to predict the label for a given audio segment
def predict(model, X,recording_count):
    
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    """
    # Add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)

    # Perform prediction
    prediction = model.predict(X)

    # Get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    predicted_index = int(predicted_index)
    print(labels_to_name[predicted_index])
    #if labels_to_name[predicted_index]=='noise':
    
    if labels_to_name[predicted_index]=='hunger' or labels_to_name[predicted_index]=='pain' or labels_to_name[predicted_index]=='normal':
        #automtate whatsapp messages
        options=webdriver.ChromeOptions()

        options.add_argument("--user-data-dir=/home/pi/.config/chromium/Default")
        browser_driver = Service('/usr/lib/chromium-browser/chromedriver')
        page = webdriver.Chrome(service=browser_driver,options=options )
        page.get("https://web.whatsapp.com/")
        wait=WebDriverWait(page,100)

        target='"NAME OF THE WHATSAPP USER"'
        message=f'The Baby is crying due to {labels_to_name[predicted_index]}'

        contact_path='//span[contains(@title,'+ target +')]'
        contact=wait.until(EC.presence_of_element_located((By.XPATH,contact_path)))
        contact.click()
        message_box_path='//*[@id="main"]/footer/div[1]/div/span[2]/div/div[2]/div[1]/div/div[1]'
        message_box=wait.until(EC.presence_of_element_located((By.XPATH,message_box_path)))
        message_box.send_keys(message + Keys.ENTER)
        sleep(0.2)

        contact2_path="//div[contains(@title,'Attach')]"
        contact2=wait.until(EC.presence_of_element_located((By.XPATH,contact2_path)))
        contact2.click()
        contact3_path="//input[contains(@accept,'image/*,video/mp4,video/3gpp,video/quicktime')]"
        
        picam2 = Picamera2()
        picam2.start_and_capture_file("/home/pi/test0.jpg")
        picam2.close()
        
        contact3=wait.until(EC.presence_of_element_located((By.XPATH,contact3_path)))
        contact3.send_keys("/home/pi/test0.jpg")
        sleep(3)
        contact4_path="//span[contains(@data-icon,'send')]"
        contact4=wait.until(EC.presence_of_element_located((By.XPATH,contact4_path)))
        contact4.click()
        sleep(3)
        
        contact5_path="//div[contains(@title,'Attach')]"
        contact5=wait.until(EC.presence_of_element_located((By.XPATH,contact2_path)))
        contact5.click()
        contact3_path="//input[contains(@accept,'*')]"

        contact3=wait.until(EC.presence_of_element_located((By.XPATH,contact3_path)))
        contact3.send_keys(f"/home/pi/Downloads/project/for pi/recordings/recording_{recording_count}.wav")
        sleep(3)
        contact4_path="//span[contains(@data-icon,'send')]"
        contact4=wait.until(EC.presence_of_element_located((By.XPATH,contact4_path)))
        contact4.click()
        sleep(15)
        page.quit()
        os.remove('/home/pi/test0.jpg')
       #cradle control
        duration = 20
        start_time = time()
        while time() - start_time < duration:
            GPIO.output(in4,GPIO.HIGH)
            GPIO.output(in3,GPIO.LOW)
            sleep(1)
            GPIO.output(in4,GPIO.LOW)
            GPIO.output(in3,GPIO.LOW)
            
            GPIO.output(in3,GPIO.HIGH)
            GPIO.output(in4,GPIO.LOW)
            sleep(1)
            GPIO.output(in4,GPIO.LOW)
            GPIO.output(in3,GPIO.LOW)
        GPIO.cleanup()
                
# Function to split audio into segments and perform prediction
def split_audio_and_predict(input_file, recording_count,segment_duration=3):
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

        # Compute MFCC features for the segment
        mfcc = feature.mfcc(y=segment, sr=sr, n_mfcc=23, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        mfcc = mfcc[..., np.newaxis]

        # Predict the label for the segment
        predict(model, mfcc,recording_count)

# Function to continuously record audio, save it, and predict labels
def record_and_predict(output_folder, segment_duration=3, fs=22050):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    recording_count = 1
    while True:
        # Record audio
        print(f"Recording {recording_count}...")
        recording = sd.rec(int(segment_duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        # Save recording as a WAV file
        filename = f"recording_{recording_count}.wav"
        filepath = os.path.join(output_folder, filename)
        write(filepath, fs, recording)
        print(f"Saved recording {recording_count} as {filename}")

        # Perform prediction on the saved audio segment
        print("Predicting labels for the recorded segment...")
        split_audio_and_predict(filepath,recording_count)
        recording_count += 1

# Set the input path and output folder
input_path =''
output_folder = 'recordings'

# Record audio, save it, and predict labels
record_and_predict(output_folder)
