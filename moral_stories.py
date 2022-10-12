import pandas as pd
import numpy as np
from tqdm import tqdm

def load_action_norm_split(path):
    train, dev, test = [pd.read_json(f"{path}{x}.jsonl", lines=True) for x in ["train", "dev", "test"]]
    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"
    return train, dev, test

def load_social_chem101():
    a = pd.read_csv("data/social-chem-101/social-chem-101.v1.0.tsv", sep="\t")
    return a

def load_soc101_ms_subset():
    '''
    Loads the part of social chemistry 101 restricted to the settings used by the MS authors
    '''
    social_chem = load_social_chem101()
    social_chem = social_chem[social_chem["split"] == "train"]
    social_chem = social_chem.dropna(subset=["rot-categorization", "rot-judgment", "action", "rot-agree", "action-moral-judgment"])
    social_chem = social_chem[social_chem["rot-agree"] >= 3.0]
    social_chem = social_chem[social_chem["rot-bad"] == 0]
    social_chem = social_chem[social_chem["rot-categorization"].apply(lambda x: "morality-ethics" in x or "social-norms" in x)]
    social_chem = social_chem[social_chem["rot-judgment"].apply(lambda x: "{" not in x)]
    return social_chem

def load_ms_soc_joined():
    '''
    Loads Moral Stories joined by Social Chemistry columns.
    '''
    social_chem = load_soc101_ms_subset()

    train, dev, test = load_action_norm_split("data/contrastive_moral_stories/original_ms/action+norm/norm_distance/")
    moral_stories = pd.concat([train, dev, test])

    ms = moral_stories.drop_duplicates("norm")

    mssc = ms.merge(social_chem.drop("split", axis=1), left_on="norm", right_on="rot")
    print("After joining, we retain", len(mssc["norm"].unique()), f"norms from Moral Stories ({len(ms)})")
    groups = mssc.groupby("norm", as_index=False)
    # some norms have multiple join partners, we take the first occurences
    mssc = groups.nth(0)
    mssc = mssc.rename(columns={"action":"rot-action"})
    return mssc