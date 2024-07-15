from typing import Dict, Tuple

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper
from tqdm import tqdm
from transformers import AutoTokenizer


class TransformerProcessor:

    def __init__(
            self,
            preloaded_model_name
    ):
        self.preloaded_model_name = preloaded_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.preloaded_model_name)

    def get_search_positions(
            self,
            text_ids,
            search_ids
    ):
        search_length = len(search_ids)
        found_positions = []
        for pos in range(len(text_ids)):
            if text_ids[pos: pos + search_length] == search_ids:
                found_positions.append(list(range(pos, pos + search_length)))

        return found_positions

    def process(
            self,
            texts,
            search_words
    ) -> Dict:
        text_ids = []
        text_mask = []
        search_positions = []
        search_ids = []
        search_mask = []

        for text, search_word in tqdm(zip(texts, search_words), desc='Mapping search word to text'):
            encoded_text = self.tokenizer.encode_plus(text)
            encoded_search = self.tokenizer.encode_plus(search_word, add_special_tokens=False)

            text_ids.append(encoded_text.input_ids)
            text_mask.append(encoded_text.attention_mask)
            search_ids.append(encoded_search.input_ids)
            search_mask.append(encoded_search.attention_mask)

            curr_search_positions = self.get_search_positions(text_ids=encoded_text.input_ids,
                                                              search_ids=encoded_search.input_ids)
            search_positions.append(curr_search_positions)

        return {
            'text': texts,
            'search_word': search_words,
            'text_ids': text_ids,
            'text_mask': text_mask,
            'search_positions': search_positions,
            'search_ids': search_ids,
            'search_mask': search_mask
        }


class DataProcessor:

    def __init__(
            self,
            data_loader_args,
            pad_token_id: int
    ):
        self.data_loader_args = data_loader_args
        self.pad_token_id = pad_token_id
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def batch_data(
            self,
            input_batch,
    ):
        text_ids, text_mask, sample_id, pejorative = [], [], [], []
        for batch_x, batch_y in input_batch:
            text_ids.append(th.tensor(batch_x[0], dtype=th.int32))
            text_mask.append(th.tensor(batch_x[1], dtype=th.int32))
            sample_id.append(th.tensor(batch_x[2], dtype=th.int32))

            pejorative.append(th.tensor(batch_y, dtype=th.long))

        text_ids = pad_sequence(text_ids, padding_value=self.pad_token_id, batch_first=True)
        text_mask = pad_sequence(text_mask, padding_value=0, batch_first=True)
        sample_id = th.tensor(sample_id)

        pejorative = th.tensor(pejorative)

        # input
        x = {
            'text_ids': text_ids.to(self.device),
            'text_mask': text_mask.to(self.device),
            'sample_id': sample_id.to(self.device)
        }

        return x, pejorative

    def process(
            self,
            text_ids,
            text_mask,
            sample_id,
            pejorative,
    ) -> Tuple[DataLoader, int]:
        x_th_text_ids = SequenceWrapper(text_ids)
        x_th_text_mask = SequenceWrapper(text_mask)
        x_th_id = SequenceWrapper(sample_id)
        x_th_data = x_th_text_ids.zip(x_th_text_mask, x_th_id)

        y_th_data = SequenceWrapper(pejorative)

        th_data = x_th_data.zip(y_th_data)

        th_data = DataLoader(th_data,
                             shuffle=False,
                             collate_fn=self.batch_data,
                             **self.data_loader_args)

        steps = int(np.ceil(len(text_ids) / self.data_loader_args['batch_size']))

        return th_data, steps


class AnchorProcessor:

    def __init__(
            self,
            data_loader_args,
            pad_token_id: int
    ):
        self.data_loader_args = data_loader_args
        self.pad_token_id = pad_token_id
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def batch_data(
            self,
            input_batch
    ):
        text_ids, text_mask = [], []
        for batch_x in input_batch:
            text_ids.append(th.tensor(batch_x[0], dtype=th.int32))
            text_mask.append(th.tensor(batch_x[1], dtype=th.int32))

        text_ids = pad_sequence(text_ids, padding_value=self.pad_token_id, batch_first=True)
        text_mask = pad_sequence(text_mask, padding_value=0, batch_first=True)

        # input
        x = {
            'text_ids': text_ids.to(self.device),
            'text_mask': text_mask.to(self.device),
        }

        return x

    def process(
            self,
            text_ids,
            text_mask
    ) -> Tuple[DataLoader, int]:
        x_th_text_ids = SequenceWrapper(text_ids)
        x_th_text_mask = SequenceWrapper(text_mask)
        th_data = x_th_text_ids.zip(x_th_text_mask)

        th_data = DataLoader(th_data,
                             shuffle=False,
                             collate_fn=self.batch_data,
                             **self.data_loader_args)

        steps = int(np.ceil(len(text_ids) / self.data_loader_args['batch_size']))

        return th_data, steps


class FrequencyProcessor:

    def process(
            self,
            texts,
            search_words,
            anchor_map,
            anchors
    ) -> Dict:
        frequency_info = {
            'global_frequencies': {},
            'search_frequencies': [],
            'anchors': anchors
        }
        for anchor in anchors:
            anchor_frequency = sum([1 if anchor in text.casefold() else 0 for text in texts]) / len(texts)
            frequency_info['global_frequencies'][anchor] = anchor_frequency

        for search_word in set(search_words):
            if search_word not in anchor_map:
                continue

            search_texts = [text for text, search in zip(texts, search_words) if search == search_word]
            assert len(search_texts) >= 1

            for anchor in anchor_map[search_word]:
                search_anchor_frequency = sum([1 if anchor in text.casefold() else 0 for text in search_texts]) / len(
                    search_texts)
                frequency_info['search_frequencies'].append({
                    'anchor': anchor,
                    'search_word': search_word,
                    'frequency': search_anchor_frequency
                })

        return frequency_info
