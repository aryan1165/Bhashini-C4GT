"""
This script restores an ASR model from a .nemo file, converts it to TorchScript format, and saves the TorchScript files.

Functions:
    - None

Main Code:
    - Specifies the type of ASR model (CTC or hybrid CTC-RNN-T).
    - Restores the model from a .nemo file.
    - Exports the model and its preprocessor to TorchScript format.
    - Saves the TorchScript files to the specified path.
"""

from nemo.collections.asr.models import EncDecClassificationModel, EncDecCTCModel, EncDecSpeakerLabelModel, EncDecHybridRNNTCTCModel
from nemo.utils import logging

# Specify the type of model to use and the path to the .nemo file
model_type = 'hybrid_ctc_rnnt'  # or 'ctc'
nemo_file = "models_code/Assets/hi-conformer.nemo"

if model_type == 'ctc':
    logging.info("Preparing ASR model")
    model = EncDecCTCModel.restore_from(nemo_file, map_location="cpu")
elif model_type == 'hybrid_ctc_rnnt':
    logging.info("Preparing hybrid_ctc_rnnt model")
    model = EncDecHybridRNNTCTCModel.restore_from(nemo_file, map_location="cpu")
else:
    raise NameError("Available model names are 'ctc' and 'hybrid_ctc_rnnt'")

# Define the path to save the TorchScript model
pt_file = "Path_to_save_model"
logging.info("Writing TorchScript file")

# Export the model and its preprocessor to TorchScript format
model.export(pt_file)
model.preprocessor.export(pt_file[:-3] + "_preprocess.pt")
logging.info("Successfully ported TorchScript file")
