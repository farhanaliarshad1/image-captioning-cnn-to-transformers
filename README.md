<div align="center">

# Image Captioning: Transforming Objects into Words

**From CNN-RNN models to Transformers: a review of deep-learning image captioning, with a VGG16-based experiment**

![Field](https://img.shields.io/badge/field-Computer%20Vision%20%2B%20NLP-2f6db5)
![Models](https://img.shields.io/badge/models-VGG16%20%7C%20LSTM%20%7C%20Transformer-7b4fb5)
![Data](https://img.shields.io/badge/datasets-MSCOCO%20%7C%20Flickr30k-3a9a4d)
![Type](https://img.shields.io/badge/type-seminar%20paper-d9822b)

[Read the paper (PDF)](paper/Image_Captioning_Transforming_Objects_into_Words.pdf) · [Open the notebook](code/image_captioning.ipynb)

Farhan Ali Arshad, Syed Huzaifa Ali, Tayyab Tariq
Trends in Natural Language Processing (NLP Seminar) · Universität Trier · Summer Semester 2024

</div>

---

## Overview

Image captioning lies at the intersection of computer vision and natural language processing: a model takes an
image as input and produces a textual description of the objects in it, the relations between them and their
context. Applications include accessibility for blind users, content-based image retrieval and more
sophisticated human-computer interaction (Section 1.1).

The paper reviews deep-learning image captioning developments from 2015 to 2024, covering Convolutional Neural
Networks, Recurrent Neural Networks and Transformers, and studies model architecture, data preprocessing,
training and evaluation metrics (Abstract, Sections 1.2 and 2).

**Research question (Section 1.3)**

> How can image captioning models be optimized to produce more accurate and contextually relevant descriptions?

## Architecture

![Architecture](assets/architecture.png)

| | CNN-RNN models (Section 2.2.1) | Transformer-based models (Section 2.2.2) |
|---|---|---|
| **Encoder** | A pre-trained CNN (e.g. VGGNet, ResNet) extracts high-level features, encoded into a fixed-length vector | CNN features are passed to Transformer encoders, which process all features in parallel |
| **Decoder** | An RNN, often LSTM or GRU, generates the sentence conditioned on the previously generated words | A Transformer decoder generates the caption word by word, using self-attention over the whole image and all previous words |
| **Strengths** | Fluent, grammatically correct sentences, since RNNs capture temporal dependencies between words | Captures complex relations in the image and between image and text, often giving more accurate, context-appropriate captions |
| **Weaknesses** | Focus on local dependencies; long-range relationships can be missed, leaving out crucial context | Computationally heavier than RNN-based models and need much more training data |

## Experimental setup (Section 2.3)

| Component | Details from the paper |
|---|---|
| **Datasets** | MSCOCO: 330,000 images, each annotated by five different people. Flickr30k: 30,000 images, each paired with five captions |
| **Feature extraction** | Different CNN architectures compared for their influence on caption quality; in the experiment, a VGG16 model pre-trained on ImageNet with the last fully connected layer giving a 4096-dimensional output (Section 2.4.1) |
| **Text preprocessing** | Tokenisation, lowercasing and start/end tokens added to each caption (Section 2.4.1) |
| **Data generator** | Custom generator yielding batches of image features with their text sequences (Section 2.4.1) |
| **Training** | Two stages: maximum likelihood estimation, then reinforcement learning with Self-Critical Sequence Training (SCST), which helps optimise non-differentiable metrics such as BLEU and CIDEr |
| **Evaluation metrics** | BLEU (n-gram overlap with reference captions), METEOR (adds synonymy, stemming and paraphrasing; often correlates better with human judgement), CIDEr (consensus between captions describing the same image) |

## Results (Section 2.4)

### Training and validation loss

| Epoch | Training loss | Validation loss |
|:---:|:---:|:---:|
| 1 | 4.2383 | 4.4467 |
| 2 | 2.5820 | 4.4743 |
| 3 | 1.3518 | 5.3856 |
| 4 | 0.8912 | 5.9295 |
| 5 | 0.6910 | 6.2870 |

<p align="center"><img src="assets/model_loss.png" width="85%" alt="Figure 1: model loss"></p>
<p align="center"><i>Figure 1 of the paper: model loss</i></p>

Training loss decreased from 4.2383 to 0.6910 over five epochs, while validation loss started to rise after the
second epoch and reached 6.2870, indicating overfitting (Section 2.4.2). The paper attributes the gap to a dataset
that was too small and lacked proper regularisation, or to augmentation techniques that were too simple.

### Caption prediction

Sample images were drawn from the training set and the generated captions compared with the ground truth
(Section 2.4.3):

| Image | Ground truth | Predicted caption |
|---|---|---|
| greensig1.jpg | Green signal please walk. | startseq green signal please walk endseq |
| playing50.jpg | Children at play in the park. | startseq kids are playing in the park endseq |

<p align="center"><img src="assets/caption_predictions.png" width="90%" alt="Figure 2: caption prediction"></p>
<p align="center"><i>Figure 2 of the paper: caption prediction</i></p>

## Key findings

1. **Transformers advanced the state of the art** (Sections 2.5 and 3). Moving from CNN-RNN to deep Transformer
   models significantly improved captioning accuracy and contextual quality.
2. **Attention mechanisms are central** (Sections 2.5 and 3). They let the model focus on the relevant regions of an
   image while generating each word, giving more accurate and detailed captions.
3. **Transformers come at a cost** (Section 2.2.2). They are computationally heavier than RNN-based models and
   need much more training data.
4. **The experiment overfitted** (Sections 2.4.2 and 2.4.4). Stricter dropout, L2 regularisation, early stopping
   and hyperparameter optimisation are proposed to reduce it.
5. **BLEU has limits** (Section 2.3.3). It is widely used but questionable for measuring fluency and coherence;
   METEOR often correlates better with human judgement.

## Future work (Section 2.5)

- Integrate other modalities, such as audio or video, into image captioning models.
- Reduce overfitting through regularisation and hyperparameter optimisation (Section 2.4.4).

## Code

The full experiment is in [`code/image_captioning.ipynb`](code/image_captioning.ipynb) (TensorFlow / Keras),
with its original outputs. It covers:

1. Loading and cleaning the caption file, adding `startseq` / `endseq` tokens
2. Tokenisation and an 80/20 train/validation split by image
3. VGG16 feature extraction (penultimate layer, 4096-d) for every image
4. A custom `Sequence` data generator producing next-word training pairs
5. The CNN + LSTM caption model, training and the loss curve
6. Greedy caption generation and sample predictions
7. A BLEU evaluation cell

The notebook was developed for the seminar project by the authors listed above and is maintained in this repository.

**Run it**

```bash
pip install tensorflow pandas numpy matplotlib seaborn tqdm nltk
jupyter notebook code/image_captioning.ipynb
```

Set `image_path` and the caption file path in the second and third cells to your copy of the dataset. The dataset
itself is not included in this repository.

### Notes on the notebook

These notes describe the code as published; they are not claims made in the paper.

- The notebook trains on a caption dataset of 1,600 images loaded from a local `visuallyimpair` folder (traffic
  signals, banknotes, ATMs, stairs, bus stops), with an 80/20 train/validation split by image.
- The caption model combines a Dense(256) image branch with an Embedding(256) + LSTM(256) text branch, followed
  by Dense(128) and a softmax over a 150-word vocabulary; it is trained with categorical cross-entropy and Adam for
  5 epochs. SCST is not implemented in the notebook.
- Sample predictions use `train.sample(15)`, i.e. training images, as stated in Section 2.4.3.
- The final BLEU cell loops over the DataFrame's column names rather than the test images, so its printed scores
  are not a valid evaluation and are not reported here.
- Early stopping and learning-rate scheduling are defined but commented out.

## Citation

```bibtex
@misc{arshad2024imagecaptioning,
  title  = {Image Captioning: Transforming Objects into Words},
  author = {Arshad, Farhan Ali and Ali, Syed Huzaifa and Tariq, Tayyab},
  year   = {2024},
  note   = {Seminar paper, Trends in Natural Language Processing, Universit{\"a}t Trier}
}
```

## License

The paper and figures are shared for portfolio and educational purposes. Please cite the paper if you use its
content.
