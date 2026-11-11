<div align="center">

# Image Captioning: Transforming Objects into Words

**From CNN-RNN encoder-decoders to Transformers: a review of deep-learning image captioning, with a VGG16 + LSTM experiment**

![Field](https://img.shields.io/badge/field-Computer%20Vision%20%2B%20NLP-2f6db5)
![Models](https://img.shields.io/badge/models-VGG16%20%7C%20LSTM%20%7C%20Transformer-7b4fb5)
![Data](https://img.shields.io/badge/data-MSCOCO%20%7C%20Flickr30k-3a9a4d)
![Type](https://img.shields.io/badge/type-seminar%20paper-d9822b)

[Read the paper (PDF)](paper/Image_Captioning_Transforming_Objects_into_Words.pdf)

Farhan Ali Arshad, with Syed Huzaifa Ali and Tayyab Tariq
Trends in Natural Language Processing (NLP Seminar) · Universität Trier · Summer Semester 2024

</div>

---

## Overview

Image captioning sits where computer vision meets natural language processing. A model receives an image and
writes a sentence that names the objects in it, how they relate to each other and the scene they belong to.
Good captions make images accessible to blind and low-vision users, improve content-based image search and
enable richer human-computer interaction.

This paper traces how captioning models developed between 2015 and 2024, compares the two dominant
architecture families, and trains a CNN-RNN captioner to study its behaviour in practice.

**Research question**

> How can image captioning models be optimised to produce more accurate and contextually relevant descriptions?

## Architecture

![Architecture](assets/architecture.png)

| | CNN-RNN encoder-decoder | Transformer encoder-decoder |
|---|---|---|
| **Visual encoder** | Pre-trained CNN (VGGNet, ResNet) compresses the image into one fixed-length vector | CNN features are passed as a sequence to a Transformer encoder |
| **Language decoder** | LSTM or GRU generates the caption word by word | Transformer decoder attends to every image feature and every previous word |
| **Strengths** | Fluent, grammatical sentences; light on compute and data | Captures long-range relations between objects; more accurate, context-aware captions |
| **Weaknesses** | Focuses on local dependencies and can miss scene context | Computationally heavy; needs much more training data |

## Experimental setup

| Component | Details |
|---|---|
| **Benchmarks** | MSCOCO (330,000 images, 5 human captions each) and Flickr30k (30,000 images, 5 captions each) |
| **Feature extraction** | VGG16 pre-trained on ImageNet; 4096-d output of the last fully connected layer |
| **Text preprocessing** | Tokenisation, lowercasing, `startseq` / `endseq` tokens around each caption |
| **Training** | Custom data generator feeding batches of image features and caption prefixes; maximum likelihood training, with Self-Critical Sequence Training (SCST) discussed for optimising non-differentiable metrics |
| **Evaluation metrics** | BLEU (n-gram overlap), METEOR (synonyms, stemming, paraphrase), CIDEr (consensus with reference captions) |

## Results

### Training behaviour

| Epoch | Training loss | Validation loss |
|:---:|:---:|:---:|
| 1 | 4.2383 | 4.4467 |
| 2 | 2.5820 | 4.4743 |
| 3 | 1.3518 | 5.3856 |
| 4 | 0.8912 | 5.9295 |
| 5 | 0.6910 | 6.2870 |

<p align="center"><img src="assets/model_loss.png" width="85%" alt="Training and validation loss"></p>

Training loss falls from 4.24 to 0.69, while validation loss starts rising after epoch 2 and ends at 6.29.
The widening gap shows that the model memorises the training captions instead of learning features that
generalise.

### Generated captions

| Image | Reference caption | Generated caption |
|---|---|---|
| greensig1.jpg | Green signal please walk. | green signal please walk |
| playing50.jpg | Children at play in the park. | kids are playing in the park |

<p align="center"><img src="assets/caption_predictions.png" width="90%" alt="Generated captions on assistive-scene images"></p>

The model was also applied to assistive scenes such as banknotes, ATMs, stairs, bus stops and road works.
It describes the scene type reliably ("please be careful upstairs ahead", "people are waiting at the bus stop"),
but confuses fine-grained details, for example reading a 200 euro note as a ten euro note and a 50 pound note as
twenty pounds. This matches the overfitting seen in the loss curves.

## Key findings

1. **Transformers outperform CNN-RNN models.** Self-attention relates every image region to every word, which
   gives more accurate and context-appropriate captions.
2. **Attention and strong visual features drive caption quality.** Attention lets the decoder focus on the
   relevant image region for each word it generates.
3. **Small-data training overfits quickly.** Stronger dropout, L2 regularisation, early stopping, data
   augmentation and hyperparameter tuning are needed.
4. **Automatic metrics are imperfect.** BLEU rewards n-gram overlap but captures fluency and coherence poorly;
   METEOR and CIDEr correlate better with human judgement.

## Future work

- Regularise the CNN-RNN baseline and replace VGG16 with a stronger backbone (ResNet, ViT).
- Train a Transformer captioner on the same data for a direct comparison.
- Report BLEU, METEOR and CIDEr on the MSCOCO and Flickr30k test splits.
- Add further modalities such as audio and video for richer descriptions.

## Code

The experiment code is available at <https://github.com/Tayyab304/NLP>.

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
