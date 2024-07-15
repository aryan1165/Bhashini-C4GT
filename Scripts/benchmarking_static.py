import numpy as np
from torch import Tensor
import torch
from pydub import AudioSegment
import librosa
import time
import pandas as pd
from SpeechRecognizer import SpeechRecognizer

def convert_samples_to_float32(samples):
    """
    Convert audio samples to float32 format. This function normalizes integer samples to the range [-1, 1].
    
    Parameters:
    samples (np.ndarray): Input audio samples.

    Returns:
    np.ndarray: Audio samples in float32 format.
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples

def load_audio(audio_file):
    """
    Load and preprocess an audio file. The function converts the audio to mono, resamples to 16 kHz, and converts
    samples to float32 format suitable for deep learning models.

    Parameters:
    audio_file (str): Path to the audio file.

    Returns:
    tuple: A tuple containing:
        - features (torch.Tensor): Preprocessed audio features.
        - length (torch.Tensor): Length of the audio sequence.
    """
    samples = AudioSegment.from_file(audio_file)
    sample_rate = samples.frame_rate
    target_sr = 16000
    num_channels = samples.channels

    if num_channels > 1:
        samples = samples.set_channels(1)
    samples = np.array(samples.get_array_of_samples())

    # Convert samples to float32
    samples = convert_samples_to_float32(samples)

    # Resample if necessary
    if target_sr is not None and target_sr != sample_rate:
        samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    features = torch.tensor(samples, dtype=torch.float).unsqueeze(0)
    length = torch.tensor([features.shape[1]]).long()
    return features, length

class AudioDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading audio files for deep learning models.
    """
    def __init__(self, audio_files):
        """
        Initialize the dataset with a list of audio files.

        Parameters:
        audio_files (list): List of paths to audio files.
        """
        self.audio_files = audio_files
    
    def __len__(self):
        """
        Return the number of audio files in the dataset.

        Returns:
        int: Number of audio files.
        """
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        Get the audio features and length for a given index.

        Parameters:
        index (int): Index of the audio file.

        Returns:
        tuple: A tuple containing:
            - features (torch.Tensor): Preprocessed audio features.
            - length (torch.Tensor): Length of the audio sequence.
        """
        audio_file = self.audio_files[index]
        f, fl = load_audio(audio_file)
        return f, fl

if __name__ == "__main__":
    """
    Main function to load a static quantized model, preprocess audio files, and calculate the Word Error Rate (WER)
    on a dataset.
    """
    torch.backends.quantized.engine = 'qnnpack' #mobile or arm based 

    model = torch.jit.load("Path_to_static_quantized_model.pt")
    scripted_model = torch.jit.script(model)

    # Load audio list from csv
    df = pd.read_csv("Path_to_benchmark_data")
    audio_files = df["audios"].iloc[1:].to_list()

    dataset = AudioDataset(audio_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Calculate WER on the dataset
    predictions = []
    infer_time = []

    for data, lengths in dataloader:
        with torch.no_grad():
            data = data.squeeze(0)
            lengths = lengths.squeeze(0)
            start = time.time()
            pred = scripted_model(data, lengths)
            end = time.time()
            predictions.append(pred)
            infer_time.append(end-start)
    
    print("Average inference time : ", sum(infer_time) / len(infer_time))

    df = pd.read_csv("Path_to_CSV")
    df = df.assign(static_quantised_cpu=predictions)
    print(df.head(5))
    df.to_csv("Path_to_CSV")
