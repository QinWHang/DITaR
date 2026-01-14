# DITaR
[AAAI'26 oral] Official implementation of "Unbiased Rectification for Sequential Recommender Systems under Fake Orders (DITaR)"

## Overview
We propose DITaR (Dual-view Identification and Targeted Rectification), an efficient and unbiased framework designed to protect sequential recommender systems from fake orders—covertly manipulated interactions strategically embedded within genuine user sequences. First, we construct a Dual-view Identification (DI) module that leverages cross-view discrepancies between collaborative statistical patterns and semantic associations (enhanced by LLaMA2-7B) to detect suspicious interaction anomalies. Then, to avoid the indiscriminate removal of data, we employ an Influence Function to quantify the actual impact of detected samples, filtering out only truly harmful orders for Targeted Rectification (TaR) via gradient ascent. This targeted approach allows the system to neutralize negative biases while preserving beneficial information and sequence integrity without the prohibitive cost of retraining from scratch. Extensive experiments demonstrate that DITaR achieves superior recommendation quality and computational efficiency across multiple baselines.
<img src="..\assets\DITaR.png">

## Requirments
- python == 3.10.16
- pytorch == 2.6.0
- transformers == 4.51.3
- numpy == 2.2.6
- pandas == 2.2.3
- scikit-learn == 1.6.1     

## How to run
