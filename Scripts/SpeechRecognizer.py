import torch
from torch import Tensor

class SpeechRecognizer(torch.nn.Module):
    """
    A speech recognition module that processes input speech signals and returns transcriptions.

    Attributes:
        model (torch.nn.Module): The main speech recognition model.
        preprocessor (torch.nn.Module): The preprocessing module for the input signal.
        vocab (list of str): The vocabulary list used for decoding the output of the model.
    """
    
    def __init__(self, model, preprocessor, vocab):
        """
        Initialize the SpeechRecognizer with a model, preprocessor, and vocabulary.

        Parameters:
            model (torch.nn.Module): The main speech recognition model.
            preprocessor (torch.nn.Module): The preprocessing module for the input signal.
            vocab (list of str): The vocabulary list used for decoding the output of the model.
        """
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.vocab = vocab
        self.vocab += ["<s>"]

    def forward(self, input_signal: Tensor, input_signal_length: Tensor) -> str:
        """
        Perform speech recognition on the input signal to produce a transcription.

        Parameters:
            input_signal (Tensor): The input speech signal tensor. Shape `[1, num_frames]`.
            input_signal_length (Tensor): The length of the input signal tensor.

        Returns:
            str: The resulting transcript.
        """
        # Preprocess the input signal
        f_p, fl_p = self.preprocessor(input_signal=input_signal, length=input_signal_length)
        
        # Pass the preprocessed signal through the model to obtain logits
        logits = self.model(f_p, fl_p)
        
        # Concatenate the logits for the first and last 256 elements
        logits = torch.cat([logits[:, :, 0:0+256], logits[:, :, -1:]], dim=-1)
        
        # Get the best path by taking the argmax of the logits along the last dimension
        best_path = torch.argmax(logits[0], dim=-1)  # [num_seq,]
        
        # Initialize variables for the hypothesis and previous character
        prev = ''
        hypothesis = ''
        
        # Decode the best path to form the hypothesis
        for i in best_path:
            char = self.vocab[i]
            if char == prev:
                continue
            if char == '<s>':
                prev = ''
                continue
            hypothesis += char
            prev = char
        
        # Replace special character '▁' with a space and strip leading/trailing spaces
        return hypothesis.replace('▁', ' ').strip()
