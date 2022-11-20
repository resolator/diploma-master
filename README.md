# [WIP] Research project within the scope of the master's thesis.

## Thesis
Investigation of the influence of architectural features of neural networks in the task of recognizing handwritten text without explicit segmentation on characters.

## Problem formulation
### Input
An image of a cut-out single-line handwritten text. Examples:
![iam_examples](data/iam_examples.png)

### Output
A recognized text from the input image.

### Training dataset
[IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) with [IAM-B](https://github.com/shonenkov/IAM-Splitting) splitting.

### Training conditions
- No preatrained models.
- Expand input with white pixels (32 for horizontal axis, 16 for vertical).
- Resize to the height=64 with keeping the aspect ratio.
- Other pre- and post-processing are disabled.
- Early stopping for 50 epochs without improvement on the validation.
- Adam optimizer with 5e-4 learning rate.
- Cross-entropy loss.
- Teacher rate is 0.8.
- Early stopping after 50 epochs.
- Batch size is 32 (for models with Batch Normalization layers).

More hyperparameters can be found in config (.cfg) files (like [this](scripts/train.cfg)).

## Experiments results
[Google Sheet](https://docs.google.com/spreadsheets/d/1lyGR1rrdM_5rV6hFVAG-l_qH5hY_nFo-RYbRMune5wY/edit?usp=sharing)


## Conclusions


## Full text
At the moment, the full text of the dissertation is available only in Russian (by [this](https://drive.google.com/file/d/1j2pHa8LQBd930r8wSNea7oac5xRiwgHQ/view?usp=sharing) link).

## Requirements
[ctcdecode](https://github.com/parlance/ctcdecode)

Other requirements can be found in [requirements.txt](requirements.txt).

## Usage

```bash
./train.py --config ./train.cfg --model-type seq2seq-light seq2seq-args --config-seq2seq seq2seq_light.cfg
```

## Repo structure
