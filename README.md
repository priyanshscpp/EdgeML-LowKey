# High-Accuracy Keyword Spotting on Edge

## Overview

This project aims to develop a low-power, real-time audio analysis embedded system to detect specific absolutist keywords, which can be used as markers for mental health language. The system will be designed using an Arduino Nano BLE Sense board equipped with a digital microphone, allowing for audio data collection, model training, and real-time keyword detection.

## Project Phases

### Phase 1: Data Collection
1. **Gather Audio Samples**: Record audio samples for a set of given absolutist keywords.
2. **Expand Dataset**: Integrate the gathered audio samples with the existing Speech Command dataset to create a comprehensive keyword spotting dataset.

### Phase 2: Model Training
3. **Feature Extraction**: Extract relevant features from the audio data using techniques discussed in class (e.g., MFCCs, spectrograms).
4. **Model Training**: Train a machine learning model to detect the absolutist keywords using the extracted features.
5. **Model Validation**: Validate the model to ensure it accurately identifies the keywords.

### Phase 3: Deployment and Testing
6. **Deploy Model**: Implement the trained model on the Arduino Nano BLE Sense board.
7. **Real-Time Testing**: Test the system for real-time keyword spotting and evaluate its performance.

## Dataset

The dataset will consist of:
- Audio samples of absolutist keywords recorded specifically for this project.

## Setup 

1. **Record Audio Samples**:
   - Use a recording device to collect audio samples of the following absolutist keywords: “all,” “must,” “never,” “none,”, “only”, and “silence”
   - Save these samples in the `data/keywords` directory.
   - [Open Speech Recording tool](https://tinyml.seas.harvard.edu/open_speech_recording/) can be used to record audio signals.
   - [Speech Command dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
 
3. **Integrate Dataset**:
   - Combine the recorded samples with the Speech Command dataset in the `data` directory.

## Training the Model

Train the model in the cloud using Google Colaboratory or locally using a Jupyter Notebook.

Use [model.ipynb](model.ipynb)

<table class="tfo-notebook-buttons">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />Jupyter Notebook</a>
  </td>
</table>
*Estimated Training Time: ~2 Hours.*

## Deployment on Arduino Nano BLE Sense

1. **Convert Model**:
   - Convert the trained model to a format compatible with the Arduino board (e.g., TensorFlow Lite).

2. **Deploy Model**:
   - Upload the model and the necessary code to the Arduino Nano BLE Sense board.
   - Refer to [micro_speech](micro_speech) folder.

3. **Real-Time Testing**:
   -  Fetch testing audios from [testing_audio](testing_audios) folder.
   -  Test the system for real-time keyword spotting and evaluate its performance.

## Results
<div>
  <img width="493" alt="result" src="https://github.com/dheerajkallakuri/High-Accuracy-Keyword-Spotting-on-Edge/assets/23552796/0aa8f0b9-be80-4258-88ad-21c117544e0c">
</div>

- There is an accuracy of 96% after training.
- In training unknown and silence words are also included apart from 5 words.
- The analysis of the confusion matrix revealed that the model exhibited high accuracy in recognizing certain keywords such as "all," "only," and "silence."
- However, its performance was comparatively weaker when identifying keywords like "must," "none," and "never."

## Video Demonstration

For a visual demonstration of this project, please refer to the video linked below:

[Project Video Demonstration](https://youtube.com/shorts/CLn4Z_gAVmA?feature=share)

[![Project Video Demonstration](https://img.youtube.com/vi/CLn4Z_gAVmA/0.jpg)](https://www.youtube.com/watch?v=CLn4Z_gAVmA)

## Reference

[Tensorflow Lite Micro_Speech](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/README.md#other-training-methods)
# EdgeML-LowKey
