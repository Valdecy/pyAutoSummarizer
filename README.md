# pyAutoSummarizer

pyAutoSummarizer - An Extractive and Abstractive Summarization Library Powered with Artificial Intelligence. 

## Introduction

pyAutoSummarizer is a sophisticated Python library developed to handle the complex task of text summarization, an essential component of NLP (Natural Language Processing). The library implements several advanced summarization algorithms, both extractive and abstractive. Extractive summarization algorithms focus on identifying and extracting key sentences or phrases from the original text to form the summary. Among the techniques utilized by pyAutoSummarizer are **TextRank**, **LexRank**, **LSA** (Latent Semantic Analysis), and **KL-Sum**. In the domain of deep learning, pyAutoSummarizer incorporates **BART** (Bidirectional and Auto-Regressive Transformers) and the use of **T5** (Text-to-Text Transfer Transformer) model, which is known for its versatility in handling a range of language tasks including summarization. Furthermore, pyAutoSummarizer also utilizes **PEGASUS** (Pre-training with Extracted Gap-sentences for Abstractive Summarization) and the OpenAI's GPT (Generative Pretrained Transformer), specifically the **chatGPT** model for abstractive summarization. Unlike extractive techniques, abstractive summarization involves generating new sentences, offering a summary that maintains the essence of the original text but may not use the exact wording.

pyAutoSummarizer stands out for its proficient preprocessing capabilities that pave the way for high-quality text summarization. Recognizing the importance of text normalization, the library offers a range of text cleansing and standardization features. It can convert text to **lowercase**, ensuring uniformity across the data. Additionally, it can **remove accents**, **remove special characters**, and **remove numbers**, which helps mitigate the text's noise. It also offers the functionality to **remove custom words**, enabling users to tailor their preprocessing needs. Notably, pyAutoSummarizer supports **stopwords** removal across various languages, including Arabic, Bengali, Bulgarian, Chinese, Czech, English, Finnish, French, German, Greek, Hebrew, Hind, Hungarian, Italian, Japanese, Korean, Marathi, Persia, Polish, Portuguese-br, Romanian, Russian, Slovak, Spanish, Swedish, Thai, and Ukrainian. The library provides flexibility in sentence segmentation, allowing sentences to be split based on **punctuation**, **character count**, or **word count**. 

To evaluate the quality of the summaries generated, pyAutoSummarizer integrates various metrics such as **Rouge-N**, **Rouge-L**, and **Rouge-S**, which compare the overlap of n-grams, longest common subsequence, and skip-bigram between the generated summary and the reference summary respectively. Additionally, it employs **BLEU** (Bilingual Evaluation Understudy), and **METEOR** (Metric for Evaluation of Translation with Explicit ORdering).

## Usage

1. Install
```bash
pip install pyAutoSummarizer
```

2. Try it in **Colab**:

Extractive Summarization
- Example 01: TextRank             ([ Colab Demo ](https://colab.research.google.com/drive/1m7mF4R7s6hakuVhrwymrgqNNJpTySUM4?usp=sharing#scrollTo=npuyBY596tJ5))
- Example 02: LexRank              ([ Colab Demo ](https://colab.research.google.com/drive/1gT9fV7hAE4mvwAHbfzolF6TN3TjGgJOF?usp=sharing#scrollTo=npuyBY596tJ5))
- Example 03: LSA                  ([ Colab Demo ](https://colab.research.google.com/drive/19fUslzp43_Owib9YDCb0Xfe9XZm1OKmB?usp=sharing#scrollTo=npuyBY596tJ5))
- Example 04: KL-Sum               ([ Colab Demo ](https://colab.research.google.com/drive/19zHjE0nR1GcAWi4NQmaJh1gjpqm4sqjP?usp=sharing#scrollTo=npuyBY596tJ5))
- Example 05: BART (Deep Learning) ([ Colab Demo ](https://colab.research.google.com/drive/1sAYBDQFxwlA16nBUozgE28_xZlNzUCg-?usp=sharing))
- Example 06: T5 (Deep Learning)   ([ Colab Demo ](https://colab.research.google.com/drive/1tyWu-19xA9QMrwl_kPcGJH0ZSS3r_rDZ?usp=sharing#scrollTo=npuyBY596tJ5))

Abstractive Summarization. 
- Example 01: chatGPT (Deep Learning) ([ Colab Demo ](https://colab.research.google.com/drive/1ipl6ZnyumJeuxsYelcmZEdsXDMIuM5WG?usp=sharing#scrollTo=npuyBY596tJ5)) Requires the user to have an **API key** (https://platform.openai.com/account/api-keys)
- Example 02: PEGASUS (Deep Learning) ([ Colab Demo ](https://colab.research.google.com/drive/1RWIEm9WoZBPYA_p4A1LqKnFPaXhNsQcM?usp=sharing))

## Others

- [pyBibX](https://github.com/Valdecy/pyBibX) - A Bibliometric and Scientometric Python Library Powered with Artificial Intelligence Tools
