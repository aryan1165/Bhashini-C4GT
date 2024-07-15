import numpy as np
from torch import Tensor
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from pydub import AudioSegment
import librosa
import json
from torch.quantization import get_default_qconfig, quantize_jit
from SpeechRecognizer import SpeechRecognizer

def convert_samples_to_float32(samples):
    """
    Convert audio samples to float32 format.
    
    Parameters:
    samples (np.ndarray): The audio samples.

    Returns:
    np.ndarray: The converted float32 audio samples.
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
    Load and preprocess an audio file.
    
    Parameters:
    audio_file (str): Path to the audio file.

    Returns:
    Tuple[Tensor, Tensor]: Preprocessed audio features and their lengths.
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
    A custom Dataset class for loading audio data and transcripts.
    """
    def __init__(self, audio_files, transcripts):
        """
        Initialize the dataset with a list of audio files and corresponding transcripts.

        Parameters:
        audio_files (List[str]): List of paths to audio files.
        transcripts (List[str]): List of transcripts corresponding to the audio files.
        """
        self.audio_files = audio_files
        self.transcripts = transcripts
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
        int: Number of samples.
        """
        return len(self.audio_files)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Parameters:
        index (int): Index of the sample to retrieve.

        Returns:
        Tuple[Tensor, Tensor, str]: Preprocessed audio features, their lengths, and the corresponding transcript.
        """
        audio_file = self.audio_files[index]
        transcript = self.transcripts[index]
        f, fl = load_audio(audio_file)
        return f, fl, transcript

def calibrate(model, dataloader):
    """
    Calibrate the model using the provided dataloader.
    
    Parameters:
    model (torch.nn.Module): The model to calibrate.
    dataloader (torch.utils.data.DataLoader): The dataloader providing the data for calibration.
    """
    model.eval()
    with torch.no_grad():
        for data, lengths, trans in dataloader:
            data = data.squeeze(0)
            lengths = lengths.squeeze(0)
            model(data, lengths)

if __name__ == "__main__":
    # Load the vocabulary
    vocabulary = torch.load("Path_to_vocab.pt")
    
    # Load the preprocessing model
    preprocessor = torch.jit.load("Path_to_CPU_model_preprocess.ts", map_location="cpu")
    
    # Load the main model
    scripted_model = torch.jit.load("Path_to_model.pt", map_location="cpu")

    # Set the quantized engine to 'qnnpack' for mobile or ARM-based devices
    torch.backends.quantized.engine = 'qnnpack'

    # Initialize the SpeechRecognizer model
    model = SpeechRecognizer(scripted_model, preprocessor, vocabulary)
    scripted_model = torch.jit.script(model)

    # Load audio list from JSON
    with open("Path_to_data.json") as f:
        audio_list = json.load(f)
    audio_files = [ex["audio_filepath"] for ex in audio_list]
    transcripts = [ex["text"] for ex in audio_list]

    # Create dataset and dataloader
    dataset = AudioDataset(audio_files, transcripts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Get the default quantization configuration
    qconfig = get_default_qconfig('qnnpack')

    # Quantize the model
    quantized_model = quantize_jit(
        scripted_model,
        {'': qconfig},
        calibrate,
        [dataloader]  # Pass dataloader as a list to match the function signature
    )

    # Optimize the quantized model for mobile
    optimized_model = optimize_for_mobile(quantized_model)

    # Save the optimized model
    torch.jit.save(optimized_model, "Path_to_model_static.pt")
