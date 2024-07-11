import numpy as np
from torch import Tensor
import torch
from pydub import AudioSegment
import librosa
import json
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.mobile_optimizer import optimize_for_mobile
from SpeechRecognizer import SpeechRecognizer

def convert_samples_to_float32(samples):
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
    def __init__(self, audio_files, transcripts):
        self.audio_files = audio_files
        self.transcripts = transcripts
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        transcript = self.transcripts[index]
        f, fl = load_audio(audio_file)
        return f, fl, transcript


if __name__ == "__main__":

    vocabulary = torch.load("Path_to_vocab.pt")

    preprocessor = torch.jit.load("Path_to_model_preprocess.pt", map_location="cpu")

    scripted_model = torch.jit.load("Path_to_model.pt", map_location="cpu")

    model = SpeechRecognizer(scripted_model, preprocessor, vocabulary)

    scripted_model = torch.jit.script(model)


    # Load audio list from JSON
    with open("Path_to_data.json") as f:
        audio_list = json.load(f)
    audio_files = [ex["audio_filepath"] for ex in audio_list]
    transcripts = [ex["text"] for ex in audio_list]
    # print(audio_files[0])
    audio_files = audio_files[:1]
    transcripts = transcripts[:1]
    print(audio_files)
    dataset = AudioDataset(audio_files, transcripts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    optimized_model = optimize_for_mobile(scripted_model)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for data, lengths, trans in dataloader:
                data = data.squeeze(1)
                lengths = lengths.squeeze(1)
                print("doing")
                optimized_model(data,lengths)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    prof.export_chrome_trace("trace_non_quant.json")
    