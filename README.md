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

## General idea
Let's start from the baseline. It is typical seq2seq architecure with Bahdanau attention:
![proposal_before](data/proposal_before.png)

General hypothesis is that we can skip the `Encoder RNN` part and apply the attention directly on the features map from the backbone. So the proposed architecure will look like this:
![proposal_after](data/proposal_after.png)

It this works, then we could interpret attention maps as a segmentations maps for each decoded symbol. This will be also checked in this work.

Also some secondary hypotheses will be checked.

## Experiments
### Baseline
The backbone compresses image into a single line with shape (1 x W x 256) and consists of the following layers:
```
Conv 5x5, 32 channels   -> Batch Norm -> ReLU -> Max Pool 2x2
Conv 5x5, 64 channels   -> Batch Norm -> ReLU -> Max Pool 2x2
Conv 3x3, 128 channels  -> Batch Norm -> ReLU -> Max Pool 2x1
Conv 3x3, 128 channels  -> Batch Norm -> ReLU -> Max Pool 2x1
Conv 3x3, 256 channels  -> Batch Norm -> ReLU -> Max Pool 2x1
Conv 1x1, 256 channels  -> Batch Norm -> ReLU -> Max Pool 2x1
```

The baseline seq2seq achieved `16.211` CER on the test subset (experiment name is `2022-06-14_seq2seq`).

### Replacing encoder RNN with PositionalEncoder
The first experiment is very simple: let's just remove the encoder RNN. Attention is looking on the raw output from backbone. Model can't train - CER is `75.32` (experiment name is `2022-06-16_seq2seqL_seq2seq_no_enc`). It took into account embeddings only totally ignored the visual part.

To fix that let's assume that the encoder introduced a spatial information that was critical. To check this hypothesis the PositionalEncoder layer was added between the backbone and attention layers. The resulted CER is `21.511` (experiment name is `2022-06-13_seq2seqL_LN2BN`). Looks like PE adds some some spatial information which was removed with encoder RNN but can't fully compensate it.

Comparison table:
| Architecture   | Old       | New       | Impact   |
|----------------|-----------|-----------|----------|
| Parameters num | 1,782,936 | 1,387,672 | −395,264 |
| CER            | 16.211    | 21.511    | +5.3     |
| GPU (BS=24)    | 68 ms     | 50 ms     | −26.5%   |
| CPU (BS=24)    | 4141 ms   | 3673 ms   | −11.3%   |
| CPU (BS=1)     | 281 ms    | 201 ms    | −28.5%   |

### Normalization layer
To avoid dependence on the batch size the following experiment was carried out to replace BatchNorm with InstanceNorm (experiment name is `2022-06-13_seq2seqL_LN_5e-4`).

| Norm type      | BatchNorm | InstanceNorm | Impact |
|----------------|-----------|--------------|--------|
| Parameters num | 1,387,672 | 1,384,210    | −3,462 |
| CER            | 21.511    | 19.672       | −1.839 |
| GPU (BS=24)    | 50 ms     | 52 ms        | +4%    |
| CPU (BS=24)    | 3673 ms   | 4975 ms      | +35.4% |
| CPU (BS=1)     | 201 ms    | 248 ms       | +23.4% |

In view that the normalization switch led to quality improvement the same change was applied to the baseline (experiment name is `2022-06-23_seq2seq_BN2LN`).

| Norm type      | BatchNorm | InstanceNorm | Impact |
|----------------|-----------|--------------|--------|
| Parameters num | 1,782,936 | 1,779,474    | -3,462 |
| CER            | 16.211    | 14.768       | -1.443 |
| GPU (BS=24)    | 68 ms     | 71 ms        | +4.4%  |
| CPU (BS=24)    | 4141 ms   | 5286 ms      | +27.7% |
| CPU (BS=1)     | 281 ms    | 335 ms       | +19.2% |

In the end, the relative effect is the same (-8.5% and -8.9% CER) but the processing time increased. Most likely the InstanceNorm implementation that was used is not efficient enough.

### Recurrency layer
Another small change was testes - switch from LSTM to GRU, because the last one has less parameters (experiment name is `2022-06-13_seq2seqL_gru`).

| Recurrency type | LSTM      | GRU       | Impact   |
|-----------------|-----------|-----------|----------|
| Parameters num  | 1,384,210 | 1,236,242 | -147,968 |
| CER             | 19.672    | 20.994    | +1.322   |
| GPU (BS=24)     | 52 ms     | 63 ms     | +21.2%   |
| CPU (BS=24)     | 4975 ms   | 4832 ms   | -2.9%    |
| CPU (BS=1)      | 248 ms    | 256 ms    | +3.2%    |

GRU shows worse quality so will not applied to the future experiments.

### Gates
In view that the PositionalEncoder can't fully compensate the removed encoder RNN it is necessary to find a solution. The hypothesis is that some gating mechanism was also removed with the encoder. To test that a gating mechanism from [this paper](https://arxiv.org/abs/2012.04961) was added (experiment names `2022-06-14_seq2seqL_gate_1`, `2022-06-14_seq2seqL_gate_2` and `2022-06-14_seq2seqL_gate_3`).

| Blocks num     | 0         | 1         | 2         | 3         |
|----------------|-----------|-----------|-----------|-----------|
| Parameters num | 1,384,210 | 1,518,354 | 1,652,498 | 1,786,642 |
| Impact         |           | +134,144  | +268,288  | +402,432  |
|                |           |           |           |           |
| CER            | 19.672    | 15.799    | 18.388    | 18.776    |
| Impact         |           | -3.963    | -1.374    | -0.986    |
|                |           |           |           |           |
| GPU (BS=24)    | 52 ms     | 64 ms     | 66 ms     | 69 ms     |
| Impact         |           | +23.1%    | +26.9%    | +32.7%    |
|                |           |           |           |           |
| CPU (BS=24)    | 4975 ms   | 5076 ms   | 5317 ms   | 7237 ms   |
| Impact         |           | +2%       | +6.9%     | +45.5%    |
|                |           |           |           |           |
| CPU (BS=1)     | 248 ms    | 266 ms    | 278 ms    | 285 ms    |
| Impact         |           | +7.3%     | +12.1%    | +14.9%    |

Best performance was achieved with a single gating block. This model has less parameters and works faster than the baseline but has a bit worse quality (`15.799` vs `14.768`).

### Replacing regular convolution with depthwise separable convolution
### Increasing the backbone output size
### Increasing the attention hidden size
### 2D attention



## Experiments results
[Google Sheet](https://docs.google.com/spreadsheets/d/1lyGR1rrdM_5rV6hFVAG-l_qH5hY_nFo-RYbRMune5wY/edit?usp=sharing)


## Conclusions



## Full text
At the moment, the full text of the dissertation is available only in Russian (by [this](https://drive.google.com/file/d/1j2pHa8LQBd930r8wSNea7oac5xRiwgHQ/view?usp=sharing) link).

## Requirements
All requirements can be found in [requirements.txt](requirements.txt).

## Usage
Better to run training from the `scripts/` directory. Otherwise change paths to dataset in config files (or don't use them).

To train a CTC baseline use the following command:
```bash
./train.py --config ./train.cfg --model-type baseline baseline-args --config-baseline ./baseline.cfg
```

To train a seq2seq baseline use the following command:
```bash
./train.py --config ./train.cfg --model-type seq2seq seq2seq-args --config-seq2seq ./seq2seq.cfg
```

To train a modified seq2seq with 2D attention use the following command:
```bash
./train.py --config ./train.cfg --model-type seq2seq-light seq2seq-args --config-seq2seq ./seq2seq_light.cfg
```

## Repo structure


## Gratitude
I want to express my gratitude to the people who helped and inspired me at work, namely:
- Andrey Kuroptev - for sensitive and high-quality scientific guidance;
- Andrey Upshinsky - for valuable consultation;
- Maria Eidlina - for invaluable support in experimental activities;
- Maria Yarova - for useful discussions and critical point of view.
