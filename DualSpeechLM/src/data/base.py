from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler

from transformers import CLIPTokenizer
from utils.data_utils import filter_no_caption_or_no_image, filter_no_cls_or_no_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import torch
import numpy as np
import pyarrow.parquet as pq
import io
import math
import webdataset as wds
import json
import braceexpand
import ast

from torch.utils.data import Dataset

from PIL import Image
import PIL
import logging
import os

from data.policies import CenterCropSDTransform
from torch.utils.data import default_collate

import utils.distributed as dist_utils
import pandas as pd

from utils.logging import Path

from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)

class WebDataset(object):

    def __init__(
        self,
        path,
        tokenizer,
        num_examples_to_see,
        batch_size=256,
        workers=1,
        train=True,
        resolution=512,
        filters=None,
        **kwargs,
    ):
        self.filters = filters or {}
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.workers = workers
        self.dataset = self.get_dataset(
            path,
            tokenizer=tokenizer,
            train=train,
            num_examples_to_see=num_examples_to_see,
            filters=self.filters,
        )

        self.loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,  # Shuffling done in the webdataset
            num_workers=workers,
            persistent_workers=True,
        )

        logging.info(f"Unused dataset parameters for WebDataset: {kwargs}")

    def get_dataset(self, url, tokenizer, train, num_examples_to_see, filters):
        transform = CenterCropSDTransform(center_crop=True, size=self.resolution)

        pipeline = [wds.ResampledShards(url)]

        # TODO: Currently does not support validation sampling well
        # Don't split by worker and node since we're sampling with replacement
        # if train:
        #     pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            tarfile_to_samples_nothrow,
        ])

        if train:
            pipeline.append(wds.shuffle(2000))

        pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.select(metadata_filters(filters)),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(pixel_values="jpg;png;jpeg;webp", input_ids="txt", text_raw="txt"),
            wds.map(filter_keys(set(["pixel_values", "input_ids", "text_raw"]))),
            wds.map_dict(
                pixel_values=transform,
                input_ids=lambda text: tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0],
                text_raw=lambda text: text,
            ),
            wds.batched(self.batch_size, partial=not train, collation_fn=default_collate),
        ])

        effective_batch_size = dist_utils.compute_effective_batch_size(self.batch_size)

        num_worker_batches = math.ceil(num_examples_to_see / (effective_batch_size * self.workers))

        # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)

def metadata_filters(filter_dict):

    def filter_fn(sample):
        select = True
        if "json" not in sample:
            # No 'json' in sample to use for filtering
            # - if `filter_dict`` is not empty, then we should not select this sample
            # - if `filter_dict`` is empty, it means there is no filter and thus
            # we select the sample
            return False if filter_dict else True

        db = json.loads(sample["json"])

        for param, expr in filter_dict.items():
            if param not in db:
                logging.info("Field {param} not in sample")
                return False

            param_val = db[param]

            # TODO: This allows code injection
            select = select and eval(f"{param_val}{expr}")

            # if ">" in val:
            #     threshold = float(val.split(">")[-1])
            #     select = select and (param_val > threshold)
            # elif "<" in val:
            #     threshold = float(val.split("<")[-1])
            #     select = select and (param_val < threshold)
            # else:
            #     raise ValueError("Need direction for filter threshold")

        if not select:
            logging.info(f"Field {param} not match threshold")

        return select

    return filter_fn


def filter_keys(key_set):

    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")

    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value

    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)

    return samples


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")

    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])

    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, "r").read())

    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258

    num_shards = len(shards_list)

    return total_size, num_shards
