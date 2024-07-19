# Scripts

This directory has all the scripts that were executed to implement the goal of downsizing the ASR model.

1. export_from_nemo.py - This script export the .nemo checkpoint to a torchscript model with extension .pt. Besides requirement.txt to run this file, one has to install nemo in their virtual environment. Make sure to follow the installation steps given [here](https://github.com/NVIDIA/NeMo?tab=readme-ov-file#mac-computers-with-apple-silicon). This script can also used to export .onnx file. This script produces model and model_preprocessor as torchscript modules.
2. dynamic_quant_cpu.py - This script provides SpeechRecognizer class which loads the vocabulary, model and model_preprocessor(export from 1.). Quantised engine can be modified depending upon the system. ```quantize_dynamic_jit()``` is used to perform dynamic quantisation to get the size from 520 MB to 196 MB. 
3. static_quant_cpu.py - This script perform static quantisation of the SpeechRecognizer model after calibration of test_set. ```quantize_jit``` is used to perform the static quantisation with arguments for calibrate function and dataloader. Size was down to 143 MB after this step.
4. inference_cpu.py - This script inferences the original non-quantised model and calculates WER and average inference time on the test data for evaluation. WER - 0.2182 and Avg inference time - 0.5803 sec
5. inference_dynamic.py - This script inferences the dynamically quantised model and calculates WER and average inference time on the test data for evaluation. WER - 0.2214 and Avg inference time - 0.8088 sec
6. inference_static.py - This script inferences the dynamically quantised model and calculates WER and average inference time on the test data for evaluation. WER - 0.2221 and Avg inference time - 0.8317 sec
7. profiler_cpu.py - This script creates profile to view which layer takes how much. It also exports to view in chrome://tracing.
8. profiler_static.py -  This script creates profile to view which layer takes how much. It also exports to view in chrome://tracing.
9. save_for_lite.py - This script exports the saved optimized_for_mobile model to model for lite_interpreter (.ptl). These models are then used in the demo andrioid application.
10. benchmarking_static.py - This script is used to benchmark the statically quantised models on the benchmarking dataset.
11. SpeechRecognizer.py - this script has the SpeechRecognizer class which is the basic structure of the model, it take model_preprocessor, acoustic model and its vocabulary to define the model and its forward method.