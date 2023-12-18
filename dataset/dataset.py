import os
import nltk
nltk.download('punkt')

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(data_dir):
    sentences = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            with open(os.path.join(root, filename), 'r') as file:
                data = file.read().replace('\n', ' ')
                sentences.extend(nltk.tokenize.sent_tokenize(data))

    for item in sentences:
        yield item


def get_or_build_tokenizer(tokenizer_file, data_dir, force_build_tokenizer):
    if not Path(tokenizer_file).exists() or force_build_tokenizer is 'true':
        print("Building tokenizer...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[SOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(data_dir), trainer=trainer)
        tokenizer.save(str(tokenizer_file))
    else:
        print(f"Loading tokenizer from {tokenizer_file}")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    return tokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer, data_dir, seq_len):
        self.tokenizer = tokenizer
        self.pad_token = torch.tensor([self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.data_dir = data_dir
        self.seq_len = seq_len
        data = ""
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                file_abspath = os.path.join(root, filename)
                if data != "":
                    data += " "
                with open(file_abspath, 'r') as file:
                    data += file.read().replace('\n', ' ')
        self.encoded = self.tokenizer.encode(data).ids
        del data

    def __len__(self):
        return len(self.encoded) // self.seq_len

    def __getitem__(self, idx):
        i = torch.randint(0, len(self.encoded) - self.seq_len, (1,))
        encoder_input = torch.tensor(self.encoded[i: i + self.seq_len])
        n_pad = torch.randint(0, self.seq_len, (1,))
        if n_pad > 0:
            encoder_input[-n_pad:] = self.pad_token
        # src_text is all tokens of encoder_input, except the pad tokens
        src_text = self.tokenizer.decode(encoder_input[0: self.seq_len - n_pad].numpy().tolist())

        label = torch.tensor(self.encoded[i + 1: i + self.seq_len + 1])
        if n_pad > 0:
            label[-n_pad:] = self.pad_token
        # tgt_text is the last token of label, except the pad tokens
        tgt_text = self.tokenizer.decode(label[self.seq_len - n_pad - 1].numpy())

        return {
            'encoder_input': encoder_input,  # (seq_len,)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            'label': label, # (seq_len,)
            'src_text': src_text,
            'tgt_text': tgt_text
        }

