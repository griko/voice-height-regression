# Height Estimation Model

This model combines the SpeechBrain ECAPA-TDNN speaker embedding model with an SVR regressor to predict speaker height from audio input. The model was trained on the VoxCeleb2 and evaluated on the VoxCeleb2 and TIMIT datasets.

## Model Details
- **Architecture**: SpeechBrain ECAPA-TDNN embeddings (192-dim) + SVR regressor
  - Output: Predicted height in centimeters (continuous value)
- **Training Data**:
  - The height data was gained by querying the height parameter of VoxCeleb1 in conjunction with VoxCeleb2 from Wikidata and converted it to centimeters. 
  - It contains 1715 persons with height information for both datasets (VoxCeleb1 and VoxCeleb2), 1621 of which are present in VoxCeleb2.
  - The code and data can be found in `src\voxceleb_height_data_collection`.
  - The original VOXCELEB ENRICHMENT FOR AGE AND GENDER RECOGNITION dataset can be found [here](https://github.com/hechmik/voxceleb_enrichment_age_gender).
- **Performance**: 
  - VoxCeleb2 test set: 6.01 cm Mean Absolute Error (MAE)
  - TIMIT test set: 6.02 cm Mean Absolute Error (MAE)
- **Audio Processing**:
  - Input format: Any audio file format supported by soundfile
  - Automatically converted to: 16kHz, mono, single channel, 256 Kbps


## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/griko/voice-height-regression.git
```

## Usage

```python
from voice_height_regression import HeightRegressionPipeline

# Load the pipeline
regressor = HeightRegressionPipeline.from_pretrained(
    "griko/height_reg_svr_ecapa_voxceleb"
)

# Single file prediction
result = regressor("path/to/audio.wav")
print(f"Predicted height: {result[0]:.1f} cm")

# Batch prediction
results = regressor(["audio1.wav", "audio2.wav"])
print(f"Predicted heights: {[f'{h:.1f}' for h in results]} cm")
```

## Limitations
- Model was trained on celebrity voices from YouTube interviews
- Performance may vary on:
  - Different audio qualities
  - Different recording conditions
  - Multiple simultaneous speakers

## Citation
If you use this model in your research, please cite:
```bibtex
@misc{koushnir2025vanpyvoiceanalysisframework,
      title={VANPY: Voice Analysis Framework}, 
      author={Gregory Koushnir and Michael Fire and Galit Fuhrmann Alpert and Dima Kagan},
      year={2025},
      eprint={2502.17579},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.17579}, 
}
```
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- VoxCeleb2 dataset for providing the training data
- SpeechBrain team for their excellent speech processing toolkit
