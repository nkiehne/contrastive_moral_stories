# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

template ="""Transform the following statements to their opposite:

You should never try to be someone you're not.
You should try to become someone you're not.

It is unsafe to drive too fast.
It is good to drive too fast.

It's good to help a widow find a therapist.
It's wrong to help a widow find a therapist.

It's good to do things for your mother.
It's bad to do things for your mother.

It's mean to tell people they're not good at something.
It's nice to tell people they're not good at something.

It is responsible to look after your grandchildren.
It is irresponsible to look after your grandchildren.

It's expected that you help out with your stepchild.
It's not expected that you help out with your stepchild.

It is important to not offend others.
It is important to offend others.

You do not have to deal with another person's problems.
You have to deal with another person's problems.

It's rude not to tip your server at a restaurant.
It's polite to not tip your server at a restaurant.

It's wrong to let sports interfere with your marriage.
It's right to let sports interfere with your marriage.

It is important to not betray a friend's trust.
It is important to betray a friend's trust.

Calling another person fat is rude and disrespectful.
Calling another person fat is nice and respectful.

{prompt}
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from moral_stories import load_action_norm_split

def flip_ms(generator, max_batch_size, temperature, top_p):
    train, dev, test = load_action_norm_split("data/contrastive_moral_stories/original_ms/action+norm/norm_distance/")
    
    def flip(data):
        norms = data[["norm"]].drop_duplicates()
        prompts = [template.format(prompt=x) for x in norms["norm"]]
        flipped_norms = []
        for batch in tqdm(np.array_split(prompts, len(prompts)//max_batch_size + 1)):
            batch = batch.tolist()
            results = generator.generate(batch, max_gen_len=128, temperature=temperature, top_p=top_p, stop_ids=[13])
            results = [x.split("\n")[-2] for x in results]
            flipped_norms += results
        norms["flipped-rot"] = flipped_norms
        data = norms.merge(data, left_on="norm", right_on="norm")
        #flip moral and immoral actions
        data = data.rename(columns={"moral_action":"immoral_action", 
                                            "immoral_action":"moral_action",
                                            "norm": "flipped-rot",
                                            "flipped-rot": "norm",
        })
        # flip labels
        data["label"] = data["label"].apply(lambda x: int(not x))
        return data
        
    anti_train = flip(train)
    anti_dev = flip(dev)
    anti_test = flip(test)

    folder = "data/contrastive_moral_stories/anti_ms_llama/action+norm/norm_distance/"
    os.makedirs(folder, exist_ok=True)

    with open(folder+"train.jsonl", "w") as f:
        f.write(anti_train.to_json(orient="records", lines=True))

    
    with open(folder+"dev.jsonl", "w") as f:
        f.write(anti_dev.to_json(orient="records", lines=True))

    with open(folder+"test.jsonl", "w") as f:
        f.write(anti_test.to_json(orient="records", lines=True))
    
            


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.5,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        
    flip_ms(generator, max_batch_size, temperature, top_p)


if __name__ == "__main__":
    fire.Fire(main)
