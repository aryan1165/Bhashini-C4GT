from nemo.collections.asr.models import EncDecClassificationModel, EncDecCTCModel, EncDecSpeakerLabelModel, EncDecHybridRNNTCTCModel
from nemo.utils import logging

model_type = 'hybrid_ctc_rnnt' # or ctc 
nemo_file = "models_code/Assets/hi-conformer.nemo"
if model_type == 'ctc':
        logging.info("Preparing ASR model")
        model = EncDecCTCModel.restore_from(nemo_file, map_location="cpu")
elif model_type == 'hybrid_ctc_rnnt':
        logging.info("Preparing hybrid_ctc_rnnt model")
        model = EncDecHybridRNNTCTCModel.restore_from(nemo_file, map_location="cpu")
else:
        raise NameError("Available model names are asr, speech_label and speaker")

pt_file = "Path_to_save_model"
logging.info("Writing torchscript file")
model.export(pt_file)
model.preprocessor.export(pt_file[:-3]+"_preprocess.pt")
logging.info("succesfully ported torchscript file")