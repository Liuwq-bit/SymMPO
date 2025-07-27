# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
import logging
import pathlib
import getpass
import transformers

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from ..beit_utils import is_main_process, get_rank
from ..diff_lib import get_diff_ids, color_print_diff_pair, split_into_words
from muffin.eval.muffin_inference_logp import preference_collator_fn, concate_pad

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta_1: float
    beta_2: float
    lamda: float
    mod_token_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)

        instances_2, instances_w, instances_l = list(zip(*instances))

        batch['beta_1'] = self.beta_1
        batch['beta_2'] = self.beta_2
        batch['lamda'] = self.lamda

        for im_key in ['1', '2']:
            batch[f'ref_logp_{im_key}_w'] = torch.as_tensor([x[f'ref_logp_{im_key}_w'] for x in instances_w])
            batch[f'ref_logp_{im_key}_2'] = torch.as_tensor([x[f'ref_logp_{im_key}_2'] for x in instances_2])
            batch[f'ref_avg_logp_{im_key}_w'] = torch.as_tensor([x[f'ref_avg_logp_{im_key}_w'] for x in instances_w])
            batch[f'ref_avg_logp_{im_key}_2'] = torch.as_tensor([x[f'ref_avg_logp_{im_key}_2'] for x in instances_2])

            locals()[f'ref_per_token_logp_{im_key}_w'] = [torch.as_tensor(x[f'ref_per_token_logp_{im_key}_w']) for x in instances_w]
            locals()[f'ref_per_token_logp_{im_key}_2'] = [torch.as_tensor(x[f'ref_per_token_logp_{im_key}_2']) for x in instances_2]
            batch[f'ref_per_token_logp_{im_key}_w'] = torch.nn.utils.rnn.pad_sequence(locals()[f'ref_per_token_logp_{im_key}_w'], batch_first=True, padding_value=0)
            batch[f'ref_per_token_logp_{im_key}_2'] = torch.nn.utils.rnn.pad_sequence(locals()[f'ref_per_token_logp_{im_key}_2'], batch_first=True, padding_value=0)

            if im_key == "1":
                batch[f'ref_logp_{im_key}_l'] = torch.as_tensor([x[f'ref_logp_{im_key}_l'] for x in instances_l])
                batch[f'ref_avg_logp_{im_key}_l'] = torch.as_tensor([x[f'ref_avg_logp_{im_key}_l'] for x in instances_l])
                locals()[f'ref_per_token_logp_{im_key}_l'] = [torch.as_tensor(x[f'ref_per_token_logp_{im_key}_l']) for x in instances_l]
                batch[f'ref_per_token_logp_{im_key}_l'] = torch.nn.utils.rnn.pad_sequence(locals()[f'ref_per_token_logp_{im_key}_l'], batch_first=True, padding_value=0)


        input_ids_w = batch['input_ids_w']
        input_ids_l = batch['input_ids_l']
        input_ids_2 = batch['input_ids_2']
        labels_w = batch['labels_w']
        labels_l = batch['labels_l']
        labels_2 = batch['labels_2']

        for im_key in ['1', '2']:
            assert batch[f'ref_per_token_logp_{im_key}_w'].size(1) >= input_ids_w.size(
                1) - 1, f"{batch[f'ref_per_token_logp_{im_key}_w'].size(1)} >= {input_ids_w.size(1) - 1}"
            assert batch[f'ref_per_token_logp_{im_key}_2'].size(1) >= input_ids_2.size(
                1) - 1, f"{batch[f'ref_per_token_logp_{im_key}_2'].size(1)} >= {input_ids_2.size(1) - 1}"

            # length of logp is one-token shorter since the last token's output is not used
            batch[f'ref_per_token_logp_{im_key}_w'] = batch[f'ref_per_token_logp_{im_key}_w'][:,:input_ids_w.size(1) - 1]
            batch[f'ref_per_token_logp_{im_key}_2'] = batch[f'ref_per_token_logp_{im_key}_2'][:,:input_ids_2.size(1) - 1]

            if im_key == '1':
                assert batch[f'ref_per_token_logp_{im_key}_l'].size(1) >= input_ids_l.size(
                    1) - 1, f"{batch[f'ref_per_token_logp_{im_key}_l'].size(1)} >= {input_ids_l.size(1) - 1}"
                batch[f'ref_per_token_logp_{im_key}_l'] = batch[f'ref_per_token_logp_{im_key}_l'][:,:input_ids_l.size(1) - 1]
        #     locals()[f'token_weight_{im_key}_1'] = torch.ones_like(batch[f'ref_per_token_logp_{im_key}_1'])
        #     locals()[f'token_weight_{im_key}_2'] = torch.ones_like(batch[f'ref_per_token_logp_{im_key}_2'])

        # for idx, (w, r) in enumerate(zip(input_ids_1, input_ids_2)):
        #     valid_w = w[1:]
        #     valid_r = r[1:]
        #     min_match_size = 3
        #     r_mod, w_mod = get_diff_ids(
        #         valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
        #     r_mod_tokens = valid_r[r_mod]
        #     w_mod_tokens = valid_w[w_mod]
        #     for im_key in ['1', '2']:
        #         locals()[f'token_weight_{im_key}_1'][idx][w_mod] = self.mod_token_weight
        #         locals()[f'token_weight_{im_key}_2'][idx][r_mod] = self.mod_token_weight

        # for im_key in ['1', '2']:
        #     batch[f'token_weight_{im_key}_1'] = locals()[f'token_weight_{im_key}_1']
        #     batch[f'token_weight_{im_key}_2'] = locals()[f'token_weight_{im_key}_2']
        #     batch[f'concatenated_token_weight_{im_key}'] = concate_pad(batch[f'token_weight_{im_key}_1'], batch[f'token_weight_{im_key}_2'], 0)

        for ins in instances_w:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in instances_l:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in instances_2:
            assert len(ins['input_ids']) == len(ins['labels'])

        # for im_key in ['1', '2']:
        #     if torch.any(torch.isnan(batch[f'token_weight_{im_key}_1'])):
        #         print(f'token_weight_{im_key}_1 fail', flush=True)
        #         exit()
        #     if torch.any(torch.isnan(batch[f'token_weight_{im_key}_2'])):
        #         print(f'token_weight_{im_key}_2 fail', flush=True)
        #         exit()
        return batch
