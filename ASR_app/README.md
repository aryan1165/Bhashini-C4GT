# Speech Recognition on Android with quantised conformer model

## Introduction

This demo app showcases the Nemo conformer model and its preprocessor, which were quantised post training in static manner. the said quantised model was then optimized for mobile usage and exported for lite interpreter. 

## Prerequisites

* PyTorch 1.10.0 and torchaudio 0.10.0 (Optional)
* Python 3.8 (Optional)
* Android Pytorch library org.pytorch:pytorch_android_lite:1.10.0
* Android Studio 4.0.1 or later

## Quick Start

### 1. Get the Repo

Simply run the commands below:

```
git clone https://github.com/Bhashini-2024/Bhashini-C4GT.git
cd Bhashini-C4GT/ASR_app
```

If you don't have PyTorch 1.10.0 and torchaudio 0.10.0 installed or want to have a quick try of the demo app, then drag and drop the model file to the `app/src/main/assets` folder and continue to next Step.

### 2. Build and run with Android Studio

Start Android Studio, open the project located in `Bhashini-C4GT/ASR_app`, build and run the app on an Android device. After the app runs, tap the Start button and start saying something; after 6 seconds (you can change `private final static int AUDIO_LEN_IN_SECOND = 6;` in `MainActivity.java` for a shorter or longer recording length), the model will infer to recognize your speech. Some example recognition results are:

![](screenshot.png)
