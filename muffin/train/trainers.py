from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from ..beit_utils import is_main_process

from muffin.eval.muffin_inference_logp import get_batch_logps, get_batch_logps_minicpm


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)

class ZephyrTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        # return RandomSampler(self.train_dataset)
        return SequentialSampler(self.train_dataset)

        # if self.args.group_by_length:
        #     assert NotImplementedError
        # else:
        #     if len(self.train_dataset) >= 50_000_000:
        #         return ChunckedRandomSampler(self.train_dataset)
        #     else:
        #         # print(f'Data set size is :{len(self.train_dataset)}', flush=True)
        #         # return SequentialSampler(self.train_dataset)

        #         print(f'Shuffle Data set size is :{len(self.train_dataset)}', flush=True)
        #         return RandomSampler(self.train_dataset)

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps_minicpm if is_minicpm else get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob


def dpo_loss(policy_logps: dict,
             reference_logps: dict,
             beta_1: float,
             beta_2: float,
             lamda: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """

    ### Original DPO
    pi_logratios = policy_logps['policy_logp_1_w'] - policy_logps['policy_logp_1_l']
    ref_logratios = reference_logps['ref_logp_1_w'] - reference_logps['ref_logp_1_l']

    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta_1*logits)
    # losses = -F.logsigmoid(0.1*logits)

    ### SymMPO
    pi_logratios_1 = policy_logps['policy_logp_1_w'] - policy_logps['policy_logp_1_2']
    pi_logratios_2 = policy_logps['policy_logp_2_2'] - policy_logps['policy_logp_2_w']
    ref_logratios_1 = reference_logps['ref_logp_1_w'] - reference_logps['ref_logp_1_2']
    ref_logratios_2 = reference_logps['ref_logp_2_2'] - reference_logps['ref_logp_2_w']

    logits_1 = pi_logratios_1 - ref_logratios_1
    logits_2 = pi_logratios_2 - ref_logratios_2
    losses += lamda*(-F.logsigmoid(beta_1*logits_1) - F.logsigmoid(beta_1*logits_2))

    ## Anchor
    sft_logits_1 = policy_logps['policy_logp_1_w'] - reference_logps['ref_logp_1_w']
    sft_logits_2 = policy_logps['policy_logp_2_2'] - reference_logps['ref_logp_2_2']
    losses += -F.logsigmoid(beta_1*sft_logits_1)
    losses += -F.logsigmoid(beta_2*sft_logits_2)

    ### Regular Term
    losses += 0.0001*torch.pow(logits_1-logits_2, 2)

    rewards_1_w = beta_1 * (policy_logps['policy_logp_1_w'] - reference_logps['ref_logp_1_w']).detach()
    rewards_1_l = beta_1 * (policy_logps['policy_logp_1_l'] - reference_logps['ref_logp_1_l']).detach()
    rewards_1_2 = beta_1 * (policy_logps['policy_logp_1_2'] - reference_logps['ref_logp_1_2']).detach()
    rewards_2_w = beta_2 * (policy_logps['policy_logp_2_w'] - reference_logps['ref_logp_2_w']).detach()
    rewards_2_2 = beta_2 * (policy_logps['policy_logp_2_2'] - reference_logps['ref_logp_2_2']).detach()

    return losses, rewards_1_w, rewards_1_l, rewards_1_2, rewards_2_w, rewards_2_2

def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


def collect_preference_metrics(metrics, task,
                               rewards_1_w, rewards_1_l, rewards_1_2, rewards_2_w, rewards_2_2,
                               policy_logp_1_w, policy_logp_1_l, policy_logp_1_2, policy_logp_2_w, policy_logp_2_2,
                               ref_logp_1_w, ref_logp_1_l, ref_logp_1_2, ref_logp_2_w, ref_logp_2_2,
                               reward_accuracies, reward_accuracies_1, reward_accuracies_2,
                               preprocess_func,
                               ):
    t = task
    metrics = {}
    for im_key, conv_key in zip(['1', '1', '1', '2', '2'], ['w', 'l', '2', 'w', '2']):
        metrics[f'rewards_{t}/{im_key}-{conv_key}'] = preprocess_func(locals()[f'rewards_{im_key}_{conv_key}'])
    for im_key, conv_key in zip(['1', '1', '1', '2', '2'], ['w', 'l', '2', 'w', '2']):
        metrics[f'logps_{t}/policy_{im_key}_{conv_key}'] = preprocess_func(locals()[f'policy_logp_{im_key}_{conv_key}'])
    for im_key, conv_key in zip(['1', '1', '1', '2', '2'], ['w', 'l', '2', 'w', '2']):
        metrics[f'logps_{t}/ref_{im_key}_{conv_key}'] = preprocess_func(locals()[f'ref_logp_{im_key}_{conv_key}'])    # for im_key in ['1', '2']:
    
    metrics[f'rewards_{t}/accuracies'] = preprocess_func(reward_accuracies)
    metrics[f'rewards_{t}/accuracies_1'] = preprocess_func(reward_accuracies_1)
    metrics[f'rewards_{t}/accuracies_2'] = preprocess_func(reward_accuracies_2)
    metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/1-w'] - metrics[f'rewards_{t}/1-l']
    metrics[f'rewards_{t}/margins_1'] = metrics[f'rewards_{t}/1-w'] - metrics[f'rewards_{t}/1-2']
    metrics[f'rewards_{t}/margins_2'] = metrics[f'rewards_{t}/2-2'] - metrics[f'rewards_{t}/2-w']
    return metrics


def get_beta_and_logps(data_dict, model, args, is_minicpm=False, is_llava15=False):
    input_ids_w = data_dict.pop('input_ids_w')
    input_ids_l = data_dict.pop('input_ids_l')
    input_ids_2 = data_dict.pop('input_ids_2')

    labels_w = data_dict.pop('labels_w')
    labels_l = data_dict.pop('labels_l')
    labels_2 = data_dict.pop('labels_2')

    attention_mask_w = data_dict.pop('attention_mask_w')
    attention_mask_l = data_dict.pop('attention_mask_l')
    attention_mask_2 = data_dict.pop('attention_mask_2')

    policy, ref = dict(), dict()
    for im_key in ['1', '2']:
        if im_key == '1':
            conv_key_list = ['w', 'l', '2']
        else:
            conv_key_list = ['w', '2']
        for conv_key in conv_key_list:
            ref[f'ref_logp_{im_key}_{conv_key}'] = data_dict.pop(f'ref_logp_{im_key}_{conv_key}')
            ref[f'ref_avg_logp_{im_key}_{conv_key}'] = data_dict.pop(f'ref_avg_logp_{im_key}_{conv_key}')
            ref[f'ref_per_token_logp_{im_key}_{conv_key}'] = data_dict.pop(f'ref_per_token_logp_{im_key}_{conv_key}')
            if args.dpo_use_average:
                ref[f'ref_win_logp_{im_key}_{conv_key}'] = ref[f'ref_avg_logp_{im_key}_{conv_key}']

    beta_1 = data_dict.pop('beta_1')
    beta_2 = data_dict.pop('beta_2')
    lamda = data_dict.pop('lamda')
    if args.task == 'DPO':
        images_1 = data_dict.pop('images_1')
        images_2 = data_dict.pop('images_2')
        concatenated_images = torch.cat([images_1, images_1, images_1, images_2, images_2], dim=0)
    elif args.task == 'KTO':
        images_1 = data_dict.pop('images_1')
        images_2 = data_dict.pop('images_2')
        concatenated_images = torch.cat([images_1, images_1, images_1, images_2, images_2], dim=0)

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None

    if is_llava15:
        if args.train_lora:
            (
                _,
                _,
                _,
                _,
                concatenated_inputs_embeds,
                concatenated_labels
            ) = model.module.prepare_inputs_labels_for_multimodal(
                input_ids=concatenated_input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=concatenated_labels,
                images=concatenated_images,
            )
        else:
            (
                _,
                _,
                _,
                _,
                concatenated_inputs_embeds,
                concatenated_labels
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids=concatenated_input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=concatenated_labels,
                images=concatenated_images,
            )
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        if args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp =log_prob
    else:
        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=args.dpo_token_weighted,
                                        dpo_use_average=args.dpo_use_average,
                                        is_minicpm=is_minicpm,
                                        **data_dict)
    size_w = input_ids_w.shape[0]
    size_l = input_ids_l.shape[0]
    size_2 = input_ids_2.shape[0]
    assert size_w == size_l and size_w == size_2

    if args.dpo_token_weighted:
        if is_llava15:
            raise NotImplementedError

    policy[f'policy_logp_1_w'], policy[f'policy_logp_1_l'], policy[f'policy_logp_1_2'], \
         policy[f'policy_logp_2_w'], policy[f'policy_logp_2_2'] = concatenated_logp.split([size_w, size_l, size_2, size_w, size_2])

    return policy, ref, beta_1, beta_2, lamda



class LLaVA15DPOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_logps, ref_logps, beta_1, beta_2, lamda = get_beta_and_logps(data_dict, model, self.args, is_llava15=True)


        losses, rewards_1_w, rewards_1_l, rewards_1_2, \
            rewards_2_w, rewards_2_2 = dpo_loss(policy_logps, ref_logps, beta_1=beta_1, beta_2=beta_2, lamda=lamda)
        reward_accuracies = (rewards_1_w > rewards_1_l).float()
        reward_accuracies_1 = (rewards_1_w > rewards_1_2).float()
        reward_accuracies_2 = (rewards_2_2 > rewards_2_w).float()

        loss = losses.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, rewards_1_w, rewards_1_l, rewards_1_2, rewards_2_w, rewards_2_2,
                                             policy_logps['policy_logp_1_w'], policy_logps['policy_logp_1_l'], policy_logps['policy_logp_1_2'],
                                             policy_logps['policy_logp_2_w'], policy_logps['policy_logp_2_2'],
                                             ref_logps['ref_logp_1_w'], ref_logps['ref_logp_1_l'], ref_logps['ref_logp_1_2'],
                                             ref_logps['ref_logp_2_w'], ref_logps['ref_logp_2_2'],
                                             reward_accuracies, reward_accuracies_1, reward_accuracies_2,
                                             gather_and_do_mean)
        self.log(metrics)

        return loss
