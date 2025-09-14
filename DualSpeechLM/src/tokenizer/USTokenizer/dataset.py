

import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import torchaudio
import numpy as np
from transformers import WhisperFeatureExtractor


class USTokenizerDataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = []
        for path in ann_path:
            with open(path, "r") as f:
                data = json.load(f)
                self.annotation.extend(data["annotation"]) 

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        try:
            audio, sr = torchaudio.load(ann["path"])
            audio = audio[0]
            if sr != 16000: # resample to 16k
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
                sr = 16000
            audio = audio.numpy()
            if len(audio) < sr: # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)

            audio = audio[: sr * 30] # truncate audio to at most 30s

            whisper_extractor = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")
            spectrogram = whisper_extractor["input_features"].squeeze()

            text = ann["text"]
            task = ann.get("task", "asr")
            Q = ann.get("Q", "")

            return {
                "spectrogram": spectrogram,
                "raw_wav": audio,
                "text": text,
                "task": task,
                "Q": Q,
                "id": ann["path"],
            }
        except Exception as e:
            print(e)
            sr = 16000
            audio = np.zeros(sr, dtype=float)
            whisper_extractor = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")
            spectrogram = whisper_extractor["input_features"].squeeze()

            text = "Yes"
            task = "zerospeech_recognition"
            Q = ""

            return {
                "spectrogram": spectrogram,
                "raw_wav": audio,
                "text": text,
                "task": task,
                "Q": Q,
                "id": ann["path"],
            }
            