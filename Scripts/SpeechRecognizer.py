import torch
from torch import Tensor

class SpeechRecognizer(torch.nn.Module):
    
    def __init__(self, model, preprocessor, vocab):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.vocab = vocab
        self.vocab += ["<s>"]

    def forward(self, input_signal: Tensor, input_signal_length: Tensor) -> str:
        """Given a single channel speech data, return transcription.

        Args:
            waveforms (Tensor): Speech tensor. Shape `[1, num_frames]`.

        Returns:
            str: The resulting transcript
        """
        f_p, fl_p = self.preprocessor(input_signal=input_signal, length=input_signal_length)
        logits = self.model(f_p, fl_p)
        logits = torch.cat([logits[:, :, 0:0+256], logits[:, :, -1:]], dim=-1)
        best_path = torch.argmax(logits[0], dim=-1)  # [num_seq,]
        prev = ''
        hypothesis = ''
        for i in best_path:
            char = self.vocab[i]
            if char == prev:
                continue
            if char == '<s>':
                prev = ''
                continue
            hypothesis += char
            prev = char
        return hypothesis.replace('▁', ' ').strip()