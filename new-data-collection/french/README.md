---
task_categories:
- text-classification
language:
- fr
tags:
- hate speech
size_categories:
- 10K<n<100K
extra_gated_prompt: "You agree to not use the dataset to conduct any activity that causes harm to human subjects."
extra_gated_fields:
  Please provide more information on why you need this dataset and how you plan to use it:
    type: text
---

# French Hate Speech Superset

This dataset is a superset (N=18,071) of posts annotated as hateful or not. It results from the preprocessing and merge of all available French hate speech datasets in April 2024. These datasets were identified through a systematic survey of hate speech datasets conducted in early 2024. We only kept datasets that:
- are documented
- are publicly available
- focus on hate speech, defined broadly as "any kind of communication in speech, writing or behavior, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, color, descent, gender or other identity factor" (UN, 2019)

The survey procedure is further detailed in [our survey paper](https://aclanthology.org/2024.woah-1.23/).

## Data access and intended use
Please send an access request detailing how you plan to use the data. The main purpose of this dataset is to train and evaluate hate speech detection models, as well as study hateful discourse online. This dataset is NOT intended to train generative LLMs to produce hateful content. 

## Columns

The dataset contains six columns:
- `text`: the annotated post
- `labels`: annotation of whether the post is hateful (`== 1`) or not (`==0`). As datasets have different annotation schemes, we systematically binarized the labels.
- `target`: target of hate, in case it is provided in the raw datasets
- `source`: origin of the data (e.g., Twitter)
- `dataset`: dataset the data is from (see "Datasets" part below)
- `nb_annotators`: number of annotators by post
- `tweet_id`: tweet ID, in case the data originates from Twitter and the ID was made available.

## Datasets

The datasets that compose this superset are:
- CONAN - COunter NArratives through Nichesourcing: a Multilingual Dataset of Responses to Fight Online Hate Speech (`CONAN` in the `dataset` column)
  - [paper link](https://aclanthology.org/P19-1271/)
  - [raw data link](https://github.com/marcoguerini/CONAN)
- Multilingual and Multi-Aspect Hate Speech Analysis (`MLMA` in the `dataset` column)
  - [paper link](https://aclanthology.org/D19-1474/)
  - [raw data link](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech)
- An Annotated Corpus for Sexism Detection in French Tweets (`sexism`)
  - [paper link](https://aclanthology.org/2020.lrec-1.175/)
  - [raw data link](https://github.com/patriChiril/An-Annotated-Corpus-for-Sexism-Detection-in-French-Tweets)
- CyberAgressionAdo-v1: a Dataset of Annotated Online Aggressions in French Collected through a Role-playing Game (`cyberado`)
  - [paper link](https://hal.science/hal-03765860)
  - [raw data link](https://github.com/aollagnier/CyberAgressionAdo-v1)
- Detection of Racist Language in French Tweets (`FTR`)
  - [paper link](https://www.mdpi.com/2078-2489/13/7/318/review_report)
  - [raw data link](https://github.com/NataliaVanetik/FTR-dataset)


## Preprocessing

We drop duplicates. In case of non-binary labels, the labels are binarized (hate speech or not). We replace all usernames and links by fixed tokens to maximize user privacy. Further details on preprocessing can be found in the preprocessing code [here](https://github.com/manueltonneau/hs_geographic_survey).

## Citation
Please cite our [survey paper](https://aclanthology.org/2024.woah-1.23/) if you use this dataset.

```bibtex
@inproceedings{tonneau-etal-2024-languages,
    title = "From Languages to Geographies: Towards Evaluating Cultural Bias in Hate Speech Datasets",
    author = {Tonneau, Manuel  and
      Liu, Diyi  and
      Fraiberger, Samuel  and
      Schroeder, Ralph  and
      Hale, Scott  and
      R{\"o}ttger, Paul},
    editor = {Chung, Yi-Ling  and
      Talat, Zeerak  and
      Nozza, Debora  and
      Plaza-del-Arco, Flor Miriam  and
      R{\"o}ttger, Paul  and
      Mostafazadeh Davani, Aida  and
      Calabrese, Agostina},
    booktitle = "Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.woah-1.23",
    pages = "283--311",
    abstract = "Perceptions of hate can vary greatly across cultural contexts. Hate speech (HS) datasets, however, have traditionally been developed by language. This hides potential cultural biases, as one language may be spoken in different countries home to different cultures. In this work, we evaluate cultural bias in HS datasets by leveraging two interrelated cultural proxies: language and geography. We conduct a systematic survey of HS datasets in eight languages and confirm past findings on their English-language bias, but also show that this bias has been steadily decreasing in the past few years. For three geographically-widespread languages{---}English, Arabic and Spanish{---}we then leverage geographical metadata from tweets to approximate geo-cultural contexts by pairing language and country information. We find that HS datasets for these languages exhibit a strong geo-cultural bias, largely overrepresenting a handful of countries (e.g., US and UK for English) relative to their prominence in both the broader social media population and the general population speaking these languages. Based on these findings, we formulate recommendations for the creation of future HS datasets.",
}

```

