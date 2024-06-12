# Automated-Response-and-classification-of-infant-cries-based-on-deep-learning-techniques
## Abstract
This project aims on creating a baby monitoring system which is able to classify the live recording samples of the audio cry into 5 different classes which are hunger,pain,laugh,normal and noise.

After it predicts the reason for an infant's cry it will send a messasge via whatsapp through the web along with a photo as well as the audio sample it has just recorded.

The automation of sending whatspp messages is done using Selenium.

The whole automation part of recording and classifying is done on a raspberry pi 4b running the latest version of Raspbian OS.

This project can also be replicated on a Windows machine with a few modifications to the code.

## Working

It mainly focuses on two Models which are CNN and RNN-LSTM respectively.

We have chosen these two models and trained them and chose the model which performs the best for our training data.
RNN-lSTM perfomed better for our data compared to CNN in real world tests primarily because it performs better on sequential data like audio files so we chose RNN-LSTM for the classification purpose.

After training the model and fine tuning the parameters, we deploy this model on the raspberry pi.

Once it detects the cry, it opens a Whatasapp web using Selenium and sends the whatsapp message along with a photo and the audio recording is sent as well.

A single shaft motor is controlled by the raspberry pi and handles the swinging of the cradle if the baby is crying in an attempt to soothe the baby.

## Dataset
The Dataset combines audio samples from various youtube playlists and websites mentioned below to create a dataaset of approx.2500 samples of 3 secs long samples for the five classes.
The audio samples are downloaded and then split using the wav file spliter.py into 3 sec long segments.
Then each of these samples are manually analyzed if it contains noise or a cry audio and filtered to improve the quality of the dataset.

The dataset also incorporates samples from Baby Chillanto Database,which is a property of the Instituto Nacional de Astrofisica Optica y Electronica – CONACYT, Mexico.

We would also like to thank Dr. Carlos A. Reyes-Garcia, Dr. Emilio Arch-Tirado and his INR-Mexico group, and Dr. Edgar M. Garcia-Tamayo for their dedication of the collection of the Infant Cry data base.

HUNGER:

[Hunger playlist-1](https://www.youtube.com/playlist?list=PL3c8pbVXDYnGT8IbavTdUo2LqdGDIHZbi)

[Hunger playlist-2](https://www.youtube.com/playlist?list=PL8ev3X5tHcOZ1cMK47KJKzyybPGg1ASdN)

[Hunger playlist-3](https://www.youtube.com/playlist?list=PLi24J1tB5dgHDG6lNbSxeHtAIte73L56w)

LAUGH:

[Laugh playlist-1](https://www.youtube.com/playlist?list=PL46077C057708485D)

[Laugh playlist-2](https://www.youtube.com/playlist?list=PL9FDDA065EED56EE5)

[Laugh playlist-3](https://www.youtube.com/playlist?list=PL17qYU0cLqMqkAN_VFN8hj2Ylxw9LLWTq)

PAIN:

[Pain playlist-1](https://www.youtube.com/playlist?list=PLAWqy92HSINYIRZA0kUrmM69x0e0GV03r)

[Pain playlist-2](https://www.youtube.com/playlist?list=PLF1_tLatgz_ywqj6HuCugg5eotD4KAqFa)

[Pain playlist-3](https://www.youtube.com/playlist?list=PL6yh-QMpd3PAmXHt1Czhex_1vFb9N6rxr)

NOISE:
These samples can be downloaded from these websites :
[Pixabay](https://pixabay.com/)
[Freesound](https://freesound.org/)


## Results






## References

This project incorporates code from the following source:

- DeepLearningForAudioWithPython by **musikalkemist **
  - Repository: [DeepLearningForAudioWithPython](https://github.com/musikalkemist/DeepLearningForAudioWithPython)
  - Licensed under the MIT License

  The Dataset also incorporates samples from the Baby Chillanto Dataset.

  The paper:(https://ieeexplore.ieee.org/abstract/document/4682484)
  
  O. F. Reyes-Galaviz, S. D. Cano-Ortiz and C. A. Reyes-García, "Evolutionary-Neural System to Classify Infant Cry Units for Pathologies Identification in Recently Born Babies," 2008 Seventh Mexican International Conference on Artificial Intelligence, Atizapan de Zaragoza, Mexico, 2008, pp. 330-335, doi: 10.1109/MICAI.2008.73.
keywords: {Pathology;Pediatrics;Genetics;Feature extraction;Mel frequency cepstral coefficient;Neural networks;Feedforward neural networks;Acoustic waves;Feeds;Propagation delay;Feature Selection;Evolutionary Strategies;Classification;Infant Cry Units;Pattern Recognition;Hybrid System}, 


