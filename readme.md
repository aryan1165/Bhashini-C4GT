# Bhashini 

## Introduction 

This project involves downsizing a offline ASR model trained on pre-trained on hindi language. Goal was to downsize the model to less than 150 MB mainiting accuracy and latency. For the detailed timeline and all experiments refer to timeline.md

## Techniques Used

Quantization was explored in depth as it can downsize a model significantly without losing on accuracy and often resulting in higher inference speeds.

## Directory structure

There are three main directories Scripts, Assets and ASR_app. Scripts have all the python files implementing all the processes right from exporting .nemo checkpoint to profiling of the quantised model. Assets contain all the models, vocabulary, samples on which the whole implementation is performed. ASR_app has the android application that was created to get the real-time feel of the inference process. Refer to each of the directories' README.md to get more familarity.