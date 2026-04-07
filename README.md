# CLARIS: Clear and Intelligible Speech from Whispered and Dysarthric Voices

[![Paper](https://img.shields.io/badge/Paper-ACM_DL-blue?style=flat-square)](#)
[![Demo](https://img.shields.io/badge/Demo-GitHub_Pages-orange?style=flat-square)](https://claris-w2s.github.io/CLARIS/)

This is the official inference code for the paper titled: "CLARIS: Clear and Intelligible Speech from Whispered and Dysarthric Voices", accepted to the ACM (Association for Computing Machinery) CHI conference on Human Factors in Computing Systems 2026.

## Key Features

- **Transformer-based S2U Architecture**: Auto-Regressive Transformer for CLARIS.
- **Beam Search Decoding**: Optimized `SequenceGenerator` for high-quality unit generation.
- **Integrated Vocoding**: Built-in support for Code HiFi-GAN to convert units directly to audio waveforms.
- **Flexible Configuration**: Dataclass-based parameter management in `params.py`.
- **Ensemble Support**: Components for model ensembling to improve inference robustness.

## Project Structure

```text
CLARIS_Inference/
├── data/                           # Dataset handling and collation logic
├── models/                         # Core model architectures (SpeechToUnitTransformer)
├── modules/                        # Transformer building blocks (Attention, Layers, etc.)
├── vocoders/                       # Vocoder implementations (CodeHiFiGAN)
├── params.py                       # Configuration and hyperparameter definitions
├── sequence_generator.py           # Beam search decoding logic
├── infer_beam.py                   # Main inference entry point for generating units
├── generate_waveform_from_code.py  # Script for unit-to-waveform conversion
└── utils.py                        # Helper functions and Dictionary management
```

## Getting Started

### Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For running optimized inference with OpenVINO or ONNX, install the additional packages:

- **OpenVINO**: `pip install openvino==2025.2.0`
- **ONNX Runtime**: `pip install onnxruntime-gpu==1.23.0`

### Model Checkpoints

Ensure you have the required model checkpoints in the root directory:
- `checkpoint.pt`: The S2U transformer model weights.
- `vocoder.pt`: The Code HiFi-GAN vocoder weights.

## Usage

### 1. Generate Units from Audio Features

Use `infer_beam.py` to perform inference on input features and generate discrete units.

```bash
python infer_beam.py
```
*Note: Ensure your input features are placed in the directory specified in `infer_beam.py` (default: `./whisper/`).*

### 2. Generate Waveforms from Units

Once you have generated the `.unit` file, use `generate_waveform_from_code.py` to synthesize audio.

```bash
python generate_waveform_from_code.py --inCodeFile predictions_whisper.unit --vocoder <path_to_vocoder_checkpoint> --resultsPath ./output_audio/
```

#### Arguments for `generate_waveform_from_code.py`:
- `--inCodeFile`: Path to the generated units file.
- `--vocoder`: Path to the HiFi-GAN vocoder checkpoint.
- `--resultsPath`: Directory to save the output `.wav` files.
- `--durPrediction`: Enable duration prediction (if supported by the model).
- `--cpu`: Force CPU inference (GPU inference not supported yet for Vocoder but CPU-only inference is fast enough).

## Configuration

The system is highly configurable via `params.py`. You can adjust:
- **Model Architecture**: Layers, embedding dimensions, attention heads, etc.
- **Generation Parameters**: Beam size, n-best hypotheses.
- **Vocoder Parameters**: Upsample rates, kernel sizes, and dilation.

## Optimized Inference

For production environments, models can be exported to ONNX and OpenVINO for significantly faster inference. The models are designed to be exportable without any changes to the core architecture.

### 1. ONNX Export
Use `export.py` to generate ONNX representations of the encoder and decoder.

```bash
python export.py
```
This produces:
- `encoder.onnx`: The speech encoder module.
- `decoder.onnx`: The unit decoder module (supporting dynamic-length autoregressive decoding).

The models are exported with **dynamic axes** for sequence lengths, allowing them to handle variable-duration speech inputs.

### 2. OpenVINO Conversion
Once the ONNX models are generated, convert them to OpenVINO IR format using `exportToOV.py`.

```bash
python exportToOV.py
```
This script saves the models in the OpenVINO format (`.xml` and `.bin`), which are optimized for execution on Intel CPUs.


## TODO

- [ ] Release code for the CLARIS mobile application.
- [ ] Provide dataset splits files for reproducibility.

---

## Acknowledgements

This project builds upon and takes inspiration from the following repositories:
- [fairseq](https://github.com/facebookresearch/fairseq): For the core Transformer and S2T architecture references.
- [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis): For unit-to-speech vocoding and data processing logic.

---