

import logging
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler
import soundfile as sf
import numpy as np
import torchaudio

from dist_utils import is_main_process, get_world_size, get_rank


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=is_train,
            num_replicas=get_world_size(),
            rank=get_rank()
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None and is_train,
        collate_fn=dataset.collater,
        drop_last=is_train,
    )

    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)

    return loader


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


# def prepare_one_sample(wav_path, tmp_pt_path, wav_processor, cuda_enabled=True):
def prepare_one_sample(wav_path, wav_processor, cuda_enabled=True):
    # audio, sr = sf.read(wav_path)
    # if len(audio.shape) == 2: # stereo to mono
    #     audio = audio[:, 0]

    # pt_data = torch.load(tmp_pt_path, map_location='cpu', weights_only=True)
    # # semantic_token = pt_data['semantic']
    # pitch_token = pt_data['pitch']
    # energy_token = pt_data['energy']


    audio, sr = torchaudio.load(wav_path)
    audio = audio[0]
    if sr != 16000: # resample to 16k
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
        sr = 16000
    audio = audio.numpy()
    if len(audio) < sr: # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)

        # pitch = torch.zeros(sr//320)
        # energy = torch.zeros(sr//320)
        # pitch[:pitch_token.shape[0]] = pitch_token
        # energy[:energy_token.shape[0]] = energy_token

        # pitch_token = pitch
        # energy_token = energy
    audio = audio[: sr * 30] # truncate audio to at most 30s

    whisper_extractor = wav_processor(audio, sampling_rate=sr, return_tensors="pt")
    spectrogram = whisper_extractor["input_features"]
    # spectrogram_attention_mask = whisper_extractor["attention_mask"]

    samples = {
        "spectrogram": spectrogram,
        # "spectrogram_attention_mask": spectrogram_attention_mask,
        "raw_wav": torch.from_numpy(audio).unsqueeze(0),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        # "pitch_label": pitch_token.unsqueeze(0),
        # "energy_label": energy_token.unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples