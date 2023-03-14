# Supplementary materials for our paper "Contextualizing Language Models for Norms Diverging from Social Majority" (Kiehne et al., 2022)
## Full paper (EMNLP Findings '22): [Here](https://aclanthology.org/2022.findings-emnlp.339/)
***

### **Data available through [here](http://www.ifis.cs.tu-bs.de/webfm_send/2491) or by cloning the repo!***
## Update v1.1: We fixed errornous generations in the `anti_ms` split. For reproducibility, the old data is still there, you can find the new version in `anti_ms_llama` folder.

\* The subfolder `original_ms` contains the `norm-distance` split of the [Moral Stories](https://github.com/demelin/moral_stories) dataset by Emelin et al.!
***

**Abstract** 
To comprehensibly contextualize decisions, artificial systems in social situations need a high
degree of awareness of the rules of conduct of
human behavior. Especially transformer-based
language models have recently been shown
to exhibit some such awareness. But what if
norms in some social setting do not adhere to
or even blatantly deviate from the mainstream?
In this paper, we introduce a novel mechanism
based on deontic logic to allow for a flexible
adaptation of individual norms by de-biasing
training data sets and a task-reduction to textual entailment. Building on the popular ’Moral
Stories’ dataset we on the one hand highlight
the intrinsic bias of current language models,
on the other hand investigate the adaptability of
pre-trained models to deviating norms in fine-tuning settings.

***

## Code
***
**Pre-requisites**

Make sure to downloade Social-Chemistry-101 dataset:
1. We require Social Chemistry 101 in "data/social-chem-101/social-chem-101.v1.0.tsv"
    * Downloadable here: https://github.com/mbforbes/social-chemistry-101

2. Our model training and experiment logging is currently outsourced to another repo. It relies on [DeepSpeed](https://github.com/microsoft/DeepSpeed) to dramatically increase training across multiple gpus.
    * `FastModelLib` needs to be installed from [here, TODO!](www.github.com)
3. Make sure to install dependencies, e.g. `pip install -r requirements.txt`.


**Running any model training**

We support two types of notebook execution:
1. Default jupyter behavior. Just run the notebooks as usual, e.g. cell-by-cell for explorative work.
2. You can use `deepspeed` and `papermill` to parametrize the training-notebooks. E.g., there is one central notebook for training the `action-classification` task, but we use papermill to adapt it to hyperparameters etc. Most of our experiments are run this way, since you can run multiple models in parallel (nice for multi-gpu nodes) and have all of them benefit from the deepspeed speedups!


**Data generation**

Steps:
1. Train a rot-generator model. Refer to `rot-generator.ipynb` and the according section in `run_model_training.ipynb`
2. Run `create_flipped_ms.ipynb` to create your own contrary norms!

**Experiments**

Note that ~6TB of disk space are needed if all model checkpoints should be kept.

`run_model_training.ipynb`: Main class for experiments. Here, all hyper-parameter configurations are deployed and evaluated.

`train_action_classification.ipynb`, `train_textual_entailment`, `train_polarity_classifier`:
* Model training procedure for action classification, textual entailment and polarity components.
* These are deployed via `run_model_training.ipynb`, but can also be run standalone (without DeepSpeed).

**Evaluation**

`rater-agreement.ipynb`: Contains the interrater agreement scores.

***
**Citation**

```
@inproceedings{kiehne-etal-2022-contextualizing,
    title = "Contextualizing Language Models for Norms Diverging from Social Majority",
    author = "Kiehne, Niklas  and
      Kroll, Hermann  and
      Balke, Wolf-Tilo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.339",
    pages = "4620--4633",
    }
```
