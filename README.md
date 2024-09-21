# Explainable Multimodal Fusion for Dementia Detection from Text and Speech

This repository contains code for the TSD 2024 paper, [Explainable Multimodal Fusion for Dementia Detection from Text and Speech](https://link.springer.com/chapter/10.1007/978-3-031-70566-3_21).

## Overview

We present our results on applying cross-modal attention to dementia detection problem, also dissect explainability of both text and speech modalities. 

<p align="center">
<img src="images/multimodal-arch.png" width=500></img>
</p>

After transforming audio into Mel-spectrogram, we encode the spectrogram with a vision transformer. The corresponding transcript is encoded with RoBERTa.


## Setup
For both architectures, cd into the directory and run `run.py`.

## Explainability 

For the text explainability part, we used good old LIME for RoBERTa. For spectrogram explainability, we used attention rollout method.


## Results at a glance
In the following LIME result, blue color indicates tokens that are indicative of the control group, while the orange color indicates tokens that are used mainly by AD patients. This transcript and spectrogram belongs to control group patient, LIME visualization points to repeated words and short words mainly. Coming to the spectrogram, ViT focused on both speech and silence parts of the audio.


<p float="left">
  <img src="images/trans.png" width="500" />
  <img src="images/spec.png" width="200" /> 
</p>




## Citation and publication
To appear in TSD 2024, citation is coming soon.



